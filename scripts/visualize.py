#!/usr/bin/env python3
"""
Knowledge Graph Visualization for LIFE AI.

Generates interactive HTML visualization showing the relationship between
Ambroxol and Parkinson's disease, including their shared mechanistic context
(GCase, alpha-synuclein, lysosome, etc.).

By default, uses BOTH Ambroxol and Parkinson's as center nodes, showing
their connection and shared neighborhood in a single unified view.

Usage:
    python scripts/visualize.py [options]

Options:
    --output, -o    Output HTML file (default: kg_visualization.html)
    --centers       Comma-separated center terms (default: "ambroxol,parkinson")
    --depth         Traversal depth from centers (default: 2)
    --max-nodes     Maximum nodes to display (default: 150)
    --full          Visualize entire graph (ignores centers/depth)
    --theme         Color theme: "medical" or "dark" (default: medical)

Examples:
    # Default: Ambroxol + Parkinson's unified view (recommended)
    python scripts/visualize.py

    # Custom centers (e.g., focus on mechanism)
    python scripts/visualize.py --centers "gcase,alpha-synuclein,lysosome"

    # Full graph
    python scripts/visualize.py --full --max-nodes 200
"""

import argparse
import asyncio
import sys
from collections import defaultdict
from pathlib import Path
from uuid import UUID

from pyvis.network import Network
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import selectinload

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import settings
from src.db.enums import EntityType, RelationLabel
from src.db.models import Entity, Evidence, Relation


# =============================================================================
# Color Schemes
# =============================================================================

MEDICAL_COLORS = {
    EntityType.DISEASE: "#e74c3c",           # Red
    EntityType.GENE: "#3498db",              # Blue
    EntityType.PROTEIN: "#9b59b6",           # Purple
    EntityType.COMPOUND_DRUG: "#2ecc71",     # Green
    EntityType.PATHWAY: "#f39c12",           # Orange
    EntityType.CELL_TYPE: "#1abc9c",         # Teal
    EntityType.TISSUE_REGION: "#e67e22",     # Dark orange
    EntityType.PHENOTYPE: "#fd79a8",         # Pink
    EntityType.BIOMARKER: "#00cec9",         # Cyan
}

DARK_COLORS = {
    EntityType.DISEASE: "#ff6b6b",
    EntityType.GENE: "#4ecdc4",
    EntityType.PROTEIN: "#a29bfe",
    EntityType.COMPOUND_DRUG: "#00b894",
    EntityType.PATHWAY: "#fdcb6e",
    EntityType.CELL_TYPE: "#81ecec",
    EntityType.TISSUE_REGION: "#fab1a0",
    EntityType.PHENOTYPE: "#ff7675",
    EntityType.BIOMARKER: "#74b9ff",
}

RELATION_COLORS = {
    RelationLabel.ASSOCIATED_WITH: "#95a5a6",
    RelationLabel.INCREASES_RISK: "#e74c3c",
    RelationLabel.DECREASES_RISK: "#27ae60",
    RelationLabel.UPREGULATES: "#e74c3c",
    RelationLabel.DOWNREGULATES: "#3498db",
    RelationLabel.ACTIVATES: "#2ecc71",
    RelationLabel.INHIBITS: "#e74c3c",
    RelationLabel.BINDS: "#9b59b6",
    RelationLabel.BIOMARKER_FOR: "#f39c12",
    RelationLabel.PART_OF: "#34495e",
    RelationLabel.EXPRESSED_IN: "#1abc9c",
}


def darken_color(hex_color: str, factor: float = 0.7) -> str:
    """Darken a hex color by a factor."""
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r, g, b = int(r * factor), int(g * factor), int(b * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def lighten_color(hex_color: str, factor: float = 1.2) -> str:
    """Lighten a hex color by a factor."""
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r, g, b = min(255, int(r * factor)), min(255, int(g * factor)), min(255, int(b * factor))
    return f"#{r:02x}{g:02x}{b:02x}"


# =============================================================================
# Graph Building Functions
# =============================================================================


async def get_dual_center_subgraph(
    db: AsyncSession,
    center_terms: list[str],
    depth: int = 2,
    max_nodes: int = 150,
) -> tuple[list[Entity], list[Relation], set[UUID], dict[UUID, int]]:
    """
    Get subgraph data expanding from multiple center terms simultaneously.
    
    This creates a unified view showing how the centers connect through
    shared neighbors and paths.
    
    Returns: (entities, relations, center_entity_ids, node_depths)
    node_depths maps entity ID -> depth level (0 = center, 1 = direct neighbor, etc.)
    """
    
    # Find all center entities matching any of the terms
    center_entities: list[Entity] = []
    for term in center_terms:
        result = await db.execute(
            select(Entity).where(Entity.normalized_name.ilike(f"%{term}%"))
        )
        found = result.scalars().all()
        center_entities.extend(found)
        if found:
            print(f"   '{term}' → {len(found)} entities: {[e.name for e in found[:3]]}")
    
    if not center_entities:
        print(f"⚠️  No entities found matching {center_terms}")
        return [], [], set(), {}
    
    # Deduplicate centers
    seen = set()
    unique_centers = []
    for e in center_entities:
        if e.id not in seen:
            seen.add(e.id)
            unique_centers.append(e)
    center_entities = unique_centers
    center_ids = {e.id for e in center_entities}
    
    print(f"   Total center entities: {len(center_entities)}")
    
    # Track depth level for each node
    node_depths: dict[UUID, int] = {e.id: 0 for e in center_entities}
    
    # BFS expanding from ALL centers simultaneously
    visited_ids: set[UUID] = set()
    frontier = set(center_ids)
    all_entities: list[Entity] = list(center_entities)
    all_relations: list[Relation] = []
    
    for current_depth in range(depth):
        if not frontier or len(visited_ids) >= max_nodes:
            break
        
        visited_ids.update(frontier)
        next_frontier: set[UUID] = set()
        
        # Get outgoing relations from frontier
        result = await db.execute(
            select(Relation)
            .where(Relation.head_id.in_(frontier))
            .options(
                selectinload(Relation.head_entity),
                selectinload(Relation.tail_entity),
            )
        )
        outgoing = result.scalars().all()
        
        # Get incoming relations to frontier
        result = await db.execute(
            select(Relation)
            .where(Relation.tail_id.in_(frontier))
            .options(
                selectinload(Relation.head_entity),
                selectinload(Relation.tail_entity),
            )
        )
        incoming = result.scalars().all()
        
        for rel in outgoing + incoming:
            all_relations.append(rel)
            
            # Add tail entity if new
            if rel.tail_entity and rel.tail_id not in visited_ids:
                if len(all_entities) < max_nodes:
                    all_entities.append(rel.tail_entity)
                    next_frontier.add(rel.tail_id)
                    if rel.tail_id not in node_depths:
                        node_depths[rel.tail_id] = current_depth + 1
            
            # Add head entity if new
            if rel.head_entity and rel.head_id not in visited_ids:
                if len(all_entities) < max_nodes:
                    all_entities.append(rel.head_entity)
                    next_frontier.add(rel.head_id)
                    if rel.head_id not in node_depths:
                        node_depths[rel.head_id] = current_depth + 1
        
        frontier = next_frontier - visited_ids
        print(f"   Depth {current_depth + 1}: {len(all_entities)} entities, {len(all_relations)} relations")
    
    # Deduplicate results
    seen_entity_ids = set()
    unique_entities = []
    for e in all_entities:
        if e.id not in seen_entity_ids:
            seen_entity_ids.add(e.id)
            unique_entities.append(e)
    
    seen_relation_ids = set()
    unique_relations = []
    for r in all_relations:
        if r.id not in seen_relation_ids:
            if r.head_id in seen_entity_ids and r.tail_id in seen_entity_ids:
                seen_relation_ids.add(r.id)
                unique_relations.append(r)
    
    return unique_entities, unique_relations, center_ids, node_depths


async def get_full_graph_data(
    db: AsyncSession,
    max_nodes: int = 200,
) -> tuple[list[Entity], list[Relation], set[UUID], dict[UUID, int]]:
    """Get full graph data (limited by max_nodes)."""
    
    # Get entities with most relations first
    result = await db.execute(
        select(Entity)
        .outerjoin(Relation, or_(
            Entity.id == Relation.head_id,
            Entity.id == Relation.tail_id
        ))
        .group_by(Entity.id)
        .order_by(func.count(Relation.id).desc())
        .limit(max_nodes)
    )
    entities = result.scalars().all()
    entity_ids = {e.id for e in entities}
    
    # Get relations between these entities
    result = await db.execute(
        select(Relation)
        .where(
            Relation.head_id.in_(entity_ids),
            Relation.tail_id.in_(entity_ids),
        )
        .options(
            selectinload(Relation.head_entity),
            selectinload(Relation.tail_entity),
        )
    )
    relations = result.scalars().all()
    
    # For full graph, all nodes are at depth 0
    node_depths = {e.id: 0 for e in entities}
    
    return entities, relations, set(), node_depths


async def get_evidence_data(db: AsyncSession) -> dict[UUID, list[dict]]:
    """Get evidence records per relation (quote text + count)."""
    result = await db.execute(
        select(Evidence.relation_id, Evidence.quote, Evidence.id)
    )
    evidence_map: dict[UUID, list[dict]] = defaultdict(list)
    for row in result.all():
        evidence_map[row[0]].append({
            "quote": row[1],
        })
    return evidence_map


# =============================================================================
# Visualization Builder
# =============================================================================


def build_network(
    entities: list[Entity],
    relations: list[Relation],
    evidence_data: dict[UUID, list[dict]],
    center_ids: set[UUID],
    node_depths: dict[UUID, int],
    theme: str = "medical",
    show_labels: bool = False,
) -> Network:
    """Build pyvis Network from graph data."""
    
    colors = MEDICAL_COLORS if theme == "medical" else DARK_COLORS
    bg_color = "#ffffff" if theme == "medical" else "#1a1a2e"
    font_color = "#333333" if theme == "medical" else "#ffffff"
    
    # Create network with adjusted dimensions to avoid panel overlap
    net = Network(
        height="100vh",
        width="100%",
        bgcolor=bg_color,
        font_color=font_color,
        directed=True,
        notebook=False,
        select_menu=False,  # Disable built-in menu (we have our own)
        filter_menu=False,  # Disable built-in filter
    )
    
    # Force-directed layout - physics runs once then stops (static nodes)
    net.set_options("""
    {
        "layout": {
            "improvedLayout": true,
            "randomSeed": 42
        },
        "physics": {
            "enabled": true,
            "barnesHut": {
                "gravitationalConstant": -25000,
                "centralGravity": 0.15,
                "springLength": 180,
                "springConstant": 0.03,
                "damping": 0.09,
                "avoidOverlap": 0.5
            },
            "stabilization": {
                "enabled": true,
                "iterations": 500,
                "updateInterval": 25,
                "fit": true
            },
            "minVelocity": 0.75
        },
        "nodes": {
            "shape": "box",
            "font": {
                "size": 11,
                "face": "Arial",
                "color": "#ffffff",
                "bold": {
                    "color": "#ffffff"
                }
            },
            "borderWidth": 2,
            "shadow": {
                "enabled": true,
                "size": 5
            },
            "margin": {
                "top": 8,
                "bottom": 8,
                "left": 10,
                "right": 10
            },
            "widthConstraint": {
                "minimum": 80,
                "maximum": 150
            }
        },
        "edges": {
            "arrows": {
                "to": {
                    "enabled": true,
                    "scaleFactor": 0.5
                }
            },
            "font": {
                "size": 9,
                "align": "middle",
                "strokeWidth": 2,
                "strokeColor": "#ffffff"
            },
            "smooth": {
                "type": "continuous",
                "roundness": 0.2
            },
            "color": {
                "inherit": false
            }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": true,
            "keyboard": {
                "enabled": true
            },
            "dragNodes": true,
            "dragView": true,
            "zoomView": true,
            "hideEdgesOnDrag": true,
            "hideEdgesOnZoom": true
        }
    }
    """)
    
    # Calculate relation counts per node (for tooltips)
    relation_counts = defaultdict(int)
    for rel in relations:
        relation_counts[rel.head_id] += 1
        relation_counts[rel.tail_id] += 1
    
    # Add nodes (box shape with label inside)
    for entity in entities:
        count = relation_counts.get(entity.id, 0)
        
        # Color based on type
        bg_color_node = colors.get(entity.type, "#7f8c8d")
        
        # Highlight center entities with gold border
        is_center = entity.id in center_ids
        if is_center:
            border_width = 4
            border_color = "#f1c40f"  # Gold
        else:
            border_width = 2
            border_color = darken_color(bg_color_node)
        
        # Build tooltip
        node_depth = node_depths.get(entity.id, 0)
        tooltip = f"<b>{entity.name}</b><br>"
        tooltip += f"Type: {entity.type.value}<br>"
        tooltip += f"Depth: {node_depth}<br>"
        tooltip += f"Connections: {count}<br>"
        if entity.aliases:
            tooltip += f"Aliases: {', '.join(entity.aliases[:3])}<br>"
        if is_center:
            tooltip += "<br><b style='color: #f1c40f;'>⭐ CENTER ENTITY</b>"
        
        # Truncate label for display
        label = entity.name[:22] + "…" if len(entity.name) > 22 else entity.name
        
        net.add_node(
            str(entity.id),
            label=label,
            title=tooltip,
            shape="box",
            color={
                "background": bg_color_node,
                "border": border_color,
                "highlight": {
                    "background": bg_color_node,
                    "border": "#f1c40f"
                },
                "hover": {
                    "background": lighten_color(bg_color_node),
                    "border": border_color
                }
            },
            font={"color": "#ffffff", "size": 11},
            borderWidth=border_width,
            margin=10,
            level=node_depth,  # Store depth for filtering
            group=entity.type.value,  # Store type for filtering
        )
    
    # Add edges (labels hidden to reduce clutter - hover to see)
    for rel in relations:
        edge_color = RELATION_COLORS.get(rel.label, "#95a5a6")
        ev_records = evidence_data.get(rel.id, [])
        ev_count = len(ev_records)
        
        width = 1 + (rel.confidence * 2) + (min(ev_count, 5) * 0.3)
        
        # Get entity names for tooltip
        head_name = next((e.name for e in entities if e.id == rel.head_id), "?")
        tail_name = next((e.name for e in entities if e.id == rel.tail_id), "?")
        
        # Build tooltip with evidence quotes
        tooltip = f"<b>{rel.label.value}</b><br>"
        tooltip += f"{head_name} → {tail_name}<br>"
        tooltip += f"Confidence: {rel.confidence:.0%}<br>"
        if ev_records:
            tooltip += f"<br><b>Evidence ({ev_count}):</b><br>"
            for i, ev in enumerate(ev_records[:3]):  # Show up to 3 quotes
                quote = ev["quote"]
                # Truncate long quotes
                if len(quote) > 150:
                    quote = quote[:147] + "..."
                # Escape HTML in quote
                quote = quote.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                tooltip += f'<i>"{quote}"</i><br>'
            if ev_count > 3:
                tooltip += f"<i>...and {ev_count - 3} more</i>"
        else:
            tooltip += "Evidence: none"
        
        edge_kwargs = {
            "title": tooltip,
            "color": edge_color,
            "width": width,
            "arrows": "to",
            "group": rel.label.value,  # Store relation type for filtering
        }
        if show_labels:
            edge_kwargs["label"] = rel.label.value
        
        net.add_edge(str(rel.head_id), str(rel.tail_id), **edge_kwargs)
    
    return net


def generate_html_report(
    net: Network,
    entities: list[Entity],
    relations: list[Relation],
    center_ids: set[UUID],
    center_terms: list[str],
    node_depths: dict[UUID, int],
    output_path: Path,
) -> None:
    """Generate HTML with graph and summary statistics."""
    
    net.save_graph(str(output_path))
    
    with open(output_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    # IMPORTANT: Export network to global scope so our filter JS can access it
    html_content = html_content.replace(
        "network = new vis.Network(container, data, options);",
        "network = new vis.Network(container, data, options);\n                    window.network = network;"
    )
    
    # Count statistics
    entity_types = defaultdict(int)
    for e in entities:
        entity_types[e.type.value] += 1
    
    relation_labels = defaultdict(int)
    for r in relations:
        relation_labels[r.label.value] += 1
    
    center_names = [e.name for e in entities if e.id in center_ids]
    title = " ↔ ".join(center_terms).title() if center_terms else "Full Graph"
    max_depth = max(node_depths.values()) if node_depths else 0
    
    # Build entity type buttons
    type_colors = {
        "Disease": "#e74c3c",
        "Compound/Drug": "#2ecc71", 
        "Gene": "#3498db",
        "Protein": "#9b59b6",
        "Pathway": "#f39c12",
        "CellType": "#1abc9c",
        "Phenotype": "#fd79a8",
        "Biomarker": "#00cec9",
        "Tissue/Region": "#e67e22",
    }
    
    type_buttons = ""
    for type_name, color in type_colors.items():
        count = entity_types.get(type_name, 0)
        if count > 0:
            short_name = type_name.replace("Compound/Drug", "Drug").replace("Tissue/Region", "Tissue").replace("CellType", "Cell")
            type_buttons += f'''<button class="type-btn active" data-type="{type_name}" style="background:{color};" onclick="toggleType('{type_name}')">{short_name} ({count})</button>\n'''
    
    # Build relation type buttons
    rel_colors = {
        "ASSOCIATED_WITH": "#95a5a6",
        "INCREASES_RISK": "#e74c3c",
        "DECREASES_RISK": "#27ae60",
        "UPREGULATES": "#c0392b",
        "DOWNREGULATES": "#2980b9",
        "ACTIVATES": "#2ecc71",
        "INHIBITS": "#e74c3c",
        "BINDS": "#9b59b6",
        "BIOMARKER_FOR": "#f39c12",
        "PART_OF": "#34495e",
        "EXPRESSED_IN": "#1abc9c",
    }
    
    rel_buttons = ""
    for rel_name, color in rel_colors.items():
        count = relation_labels.get(rel_name, 0)
        if count > 0:
            short_name = rel_name.replace("_", " ").title()
            rel_buttons += f'''<button class="rel-btn active" data-rel="{rel_name}" style="background:{color};" onclick="toggleRel('{rel_name}')">{short_name} ({count})</button>\n'''
    
    custom_html = """
    <style>
        body { margin: 0; overflow: hidden; }
        #mynetwork { position: absolute; top: 0; left: 0; right: 0; bottom: 0; }
        .panel {
            position: fixed;
            background: rgba(255,255,255,0.95);
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 10px;
            font-family: Arial, sans-serif;
            font-size: 11px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.15);
            z-index: 1000;
        }
        .panel h3 { margin: 0 0 8px 0; font-size: 13px; color: #2c3e50; }
        .panel p { margin: 3px 0; }
        
        .type-btn, .rel-btn {
            display: inline-block;
            padding: 4px 8px;
            margin: 2px;
            border: 2px solid transparent;
            border-radius: 4px;
            color: white;
            font-size: 10px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .type-btn:hover, .rel-btn:hover { transform: scale(1.05); }
        .type-btn.inactive, .rel-btn.inactive { opacity: 0.25; border-color: #999; }
        
        .depth-control {
            display: flex;
            align-items: center;
            gap: 8px;
            margin: 8px 0;
        }
        .depth-btn {
            width: 30px;
            height: 30px;
            border: 2px solid #3498db;
            background: white;
            color: #3498db;
            border-radius: 4px;
            font-size: 18px;
            cursor: pointer;
            font-weight: bold;
        }
        .depth-btn:hover { background: #3498db; color: white; }
        .depth-display {
            font-size: 14px;
            font-weight: bold;
            min-width: 90px;
            text-align: center;
        }
        .reset-btn {
            width: 100%;
            padding: 6px;
            margin-top: 8px;
            background: #34495e;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 11px;
        }
        .reset-btn:hover { background: #2c3e50; }
        .section-label { margin: 6px 0 4px 0; font-weight: bold; color: #34495e; }
        .control-scroll {
            max-height: calc(100vh - 120px);
            overflow-y: auto;
        }
    </style>
    
    <!-- Title Panel - Top Left -->
    <div class="panel" style="top: 10px; left: 10px; max-width: 220px;">
        <h3>🧬 LIFE AI Knowledge Graph</h3>
        <p style="color: #7f8c8d; margin: 0;">""" + title + """</p>
    </div>
    
    <!-- Control Panel - Top Right (scrollable) -->
    <div class="panel" style="top: 10px; right: 10px; width: 270px;">
        <h3>🎮 Controls</h3>
        <div class="control-scroll">
        
        <!-- Depth Control -->
        <p class="section-label">Depth from Centers:</p>
        <div class="depth-control">
            <button class="depth-btn" onclick="changeDepth(-1)">−</button>
            <span class="depth-display">Level <span id="depth-value">""" + str(max_depth) + """</span> / """ + str(max_depth) + """</span>
            <button class="depth-btn" onclick="changeDepth(1)">+</button>
        </div>
        
        <hr style="margin: 10px 0; border: none; border-top: 1px solid #eee;">
        
        <!-- Entity Type Filters -->
        <p class="section-label">🔵 Entity Types:</p>
        <div style="line-height: 1.8;">
        """ + type_buttons + """
        </div>
        
        <hr style="margin: 10px 0; border: none; border-top: 1px solid #eee;">
        
        <!-- Relation Type Filters -->
        <p class="section-label">🔗 Relation Types:</p>
        <div style="line-height: 1.8;">
        """ + rel_buttons + """
        </div>
        
        <button class="reset-btn" onclick="resetFilters()">🔄 Reset All Filters</button>
        </div>
    </div>
    
    <!-- Stats Panel - Bottom Left -->
    <div class="panel" style="bottom: 10px; left: 10px;">
        <b>Visible:</b> <span id="visible-nodes">""" + str(len(entities)) + """</span> nodes, <span id="visible-edges">""" + str(len(relations)) + """</span> edges
        <span style="color:#7f8c8d;"> (Total: """ + str(len(entities)) + """ / """ + str(len(relations)) + """)</span>
    </div>
    
    <!-- Help Panel - Bottom Right -->
    <div class="panel" style="bottom: 10px; right: 10px; font-size: 10px; color: #7f8c8d;">
        Drag nodes to reposition | Scroll to zoom | Hover for details | <span style="border:2px solid #f1c40f;padding:0 3px;border-radius:2px;">⭐</span> = Center
    </div>
    
    <script>
        // State
        var currentDepth = """ + str(max_depth) + """;
        var maxDepth = """ + str(max_depth) + """;
        var hiddenTypes = new Set();
        var hiddenRels = new Set();
        var centerNodeIds = new Set(""" + str([str(cid) for cid in center_ids]) + """);
        
        function waitForNetwork(callback) {
            if (typeof window.network !== 'undefined' && window.network && window.network.body) {
                callback();
            } else {
                setTimeout(function() { waitForNetwork(callback); }, 50);
            }
        }
        
        // Disable physics after stabilization so nodes stay still
        waitForNetwork(function() {
            window.network.on('stabilizationIterationsDone', function() {
                window.network.setOptions({ physics: { enabled: false } });
            });
        });
        
        function updateVisibility() {
            waitForNetwork(function() {
                var nodesDS = window.network.body.data.nodes;
                var edgesDS = window.network.body.data.edges;
                
                // PASS 1: Determine which nodes pass depth + entity type filters
                var depthTypeVisible = new Set();
                nodesDS.forEach(function(node) {
                    var nodeDepth = node.level !== undefined ? node.level : 0;
                    var nodeType = node.group;
                    var withinDepth = nodeDepth <= currentDepth;
                    var typeVisible = !hiddenTypes.has(nodeType);
                    if (withinDepth && typeVisible) {
                        depthTypeVisible.add(node.id);
                    }
                });
                
                // PASS 2: Determine which edges are visible
                //   Edge visible = both endpoints pass depth/type AND relation type not hidden
                var visibleEdges = new Set();
                var connectedNodes = new Set();  // Nodes with at least one visible edge
                var visibleEdgeCount = 0;
                
                edgesDS.forEach(function(edge) {
                    var endpointsOk = depthTypeVisible.has(edge.from) && depthTypeVisible.has(edge.to);
                    var relType = edge.group;
                    var relVisible = !hiddenRels.has(relType);
                    var shouldShow = endpointsOk && relVisible;
                    
                    visibleEdges.add(edge.id);
                    if (shouldShow) {
                        connectedNodes.add(edge.from);
                        connectedNodes.add(edge.to);
                        visibleEdgeCount++;
                    }
                });
                
                // PASS 3: Final node visibility
                //   Show if: passes depth/type AND (is center OR has visible edges)
                var finalVisible = new Set();
                var nodeUpdates = [];
                nodesDS.forEach(function(node) {
                    var passesFilters = depthTypeVisible.has(node.id);
                    var isCenter = centerNodeIds.has(node.id);
                    var hasEdges = connectedNodes.has(node.id);
                    var shouldShow = passesFilters && (isCenter || hasEdges);
                    
                    if (shouldShow) finalVisible.add(node.id);
                    nodeUpdates.push({id: node.id, hidden: !shouldShow});
                });
                nodesDS.update(nodeUpdates);
                
                // PASS 4: Update edges (only show if both endpoints are in finalVisible)
                var edgeUpdates = [];
                visibleEdgeCount = 0;
                edgesDS.forEach(function(edge) {
                    var endpointsOk = finalVisible.has(edge.from) && finalVisible.has(edge.to);
                    var relVisible = !hiddenRels.has(edge.group);
                    var shouldShow = endpointsOk && relVisible;
                    edgeUpdates.push({id: edge.id, hidden: !shouldShow});
                    if (shouldShow) visibleEdgeCount++;
                });
                edgesDS.update(edgeUpdates);
                
                document.getElementById('visible-nodes').textContent = finalVisible.size;
                document.getElementById('visible-edges').textContent = visibleEdgeCount;
            });
        }
        
        function changeDepth(delta) {
            currentDepth = Math.max(0, Math.min(maxDepth, currentDepth + delta));
            document.getElementById('depth-value').textContent = currentDepth;
            updateVisibility();
        }
        
        function toggleType(typeName) {
            var btn = document.querySelector('.type-btn[data-type="' + typeName + '"]');
            if (hiddenTypes.has(typeName)) {
                hiddenTypes.delete(typeName);
                btn.classList.remove('inactive');
                btn.classList.add('active');
            } else {
                hiddenTypes.add(typeName);
                btn.classList.remove('active');
                btn.classList.add('inactive');
            }
            updateVisibility();
        }
        
        function toggleRel(relName) {
            var btn = document.querySelector('.rel-btn[data-rel="' + relName + '"]');
            if (hiddenRels.has(relName)) {
                hiddenRels.delete(relName);
                btn.classList.remove('inactive');
                btn.classList.add('active');
            } else {
                hiddenRels.add(relName);
                btn.classList.remove('active');
                btn.classList.add('inactive');
            }
            updateVisibility();
        }
        
        function resetFilters() {
            currentDepth = maxDepth;
            document.getElementById('depth-value').textContent = currentDepth;
            hiddenTypes.clear();
            hiddenRels.clear();
            document.querySelectorAll('.type-btn, .rel-btn').forEach(function(btn) {
                btn.classList.remove('inactive');
                btn.classList.add('active');
            });
            updateVisibility();
        }
        
        // Initialize
        waitForNetwork(function() {
            updateVisibility();
        });
    </script>
    """
    
    html_content = html_content.replace("</body>", f"{custom_html}</body>")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)


# =============================================================================
# Main
# =============================================================================


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate interactive Knowledge Graph visualization"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="kg_visualization.html",
        help="Output HTML file (default: kg_visualization.html)",
    )
    parser.add_argument(
        "--centers", "-c",
        type=str,
        default="ambroxol,parkinson",
        help="Comma-separated center terms (default: ambroxol,parkinson)",
    )
    parser.add_argument(
        "--depth", "-d",
        type=int,
        default=2,
        help="Traversal depth from centers (default: 2)",
    )
    parser.add_argument(
        "--max-nodes", "-n",
        type=int,
        default=150,
        help="Maximum nodes to display (default: 150)",
    )
    parser.add_argument(
        "--full", "-f",
        action="store_true",
        help="Visualize entire graph (ignores centers/depth)",
    )
    parser.add_argument(
        "--theme", "-t",
        type=str,
        choices=["medical", "dark"],
        default="medical",
        help="Color theme (default: medical)",
    )
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Show relation labels on edges (can be cluttered)",
    )
    
    args = parser.parse_args()
    output_path = Path(args.output)
    center_terms = [t.strip() for t in args.centers.split(",")]
    
    print("🎨 LIFE AI Knowledge Graph Visualization")
    print("=" * 50)
    
    engine = create_async_engine(settings.db_url, echo=False)
    
    async with AsyncSession(engine) as db:
        if args.full:
            print(f"📊 Loading full graph (max {args.max_nodes} nodes)...")
            entities, relations, center_ids, node_depths = await get_full_graph_data(
                db, args.max_nodes
            )
        else:
            print(f"📊 Loading subgraph for: {' + '.join(center_terms)}")
            print(f"   Depth: {args.depth}, Max nodes: {args.max_nodes}")
            entities, relations, center_ids, node_depths = await get_dual_center_subgraph(
                db, center_terms, args.depth, args.max_nodes
            )
        
        print(f"\n   Final: {len(entities)} entities, {len(relations)} relations")
        
        if not entities:
            print("❌ No data to visualize!")
            return
        
        print("📝 Loading evidence data...")
        evidence_data = await get_evidence_data(db)
        
        print("🔨 Building visualization...")
        net = build_network(
            entities, relations, evidence_data,
            center_ids, node_depths,
            theme=args.theme,
            show_labels=args.show_labels,
        )
        
        print("💾 Generating HTML report...")
        generate_html_report(
            net, entities, relations, center_ids, center_terms, node_depths, output_path
        )
    
    await engine.dispose()
    
    print()
    print("=" * 50)
    print(f"✅ Visualization saved to: {output_path}")
    print()
    print("Open in browser to explore!")
    print("  • Use depth +/- to expand/collapse from centers")
    print("  • Click type buttons to show/hide entity types")
    print("  • Drag to pan, scroll to zoom")
    print("  • Hover for details")


if __name__ == "__main__":
    asyncio.run(main())