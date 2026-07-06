library(ggraph)
library(tidygraph)

plot_dim <- list(width = 1684, height = 1684 * 9/16)
dir.create('plots', showWarnings = FALSE)

sysfonts::font_add_google('Space Grotesk', "space_grotesk") 
showtext::showtext_auto()

palette <- list(
  'teal' = '#14b8a6',
  'pink' = '#ec4899',
  'grey' = '#888888'
)

# Build node dataframe
nodes_df <- data.frame(
  name  = c("x1", "x2", "bias", "noise", "h1", "h2", "h3", "bias", "out"),
  group = c(rep("input", 4), rep("hidden", 4), "output"),
  x     = c(rep(0, 4), rep(2, 4), 4),
  y     = c(3, 1, -3, -1, 3, 1, -1, -3, 0)
)

# Build edge dataframe from learned weights
# w_ih1 is (4x3): rows = input nodes, cols = hidden nodes
input_edges <- expand.grid(from_idx = 1:4, to_idx = 1:3)
input_edges$from   <- input_edges$from_idx
input_edges$to     <- input_edges$to_idx + 4
input_edges$weight <- mapply(\(r, c) w_ih1[r, c], input_edges$from_idx, input_edges$to_idx)

# w_h1o is (4x1): rows = [bias, h1, h2, h3], cols = output
# cbind(1, z1) prepends bias as column 1, so w_h1o row 1 = hidden bias (node 8)
output_edges <- data.frame(
  from   = c(8, 5, 6, 7),
  to     = 9,
  weight = w_h1o[, 1]
)

edges_df <- rbind(
  data.frame(from = input_edges$from, to = input_edges$to, weight = input_edges$weight),
  output_edges
)

edges_df$sign       <- ifelse(edges_df$weight > 0, "+ve", "-ve")
edges_df$abs_weight <- abs(edges_df$weight) / max(abs(edges_df$weight))

# Build tidygraph object and set manual layout
tg <- tidygraph::tbl_graph(nodes = nodes_df, edges = edges_df, directed = TRUE)

layout <- tg %>%
  activate(nodes) %>%
  create_layout(layout = "manual", x = .$x, y = .$y)

ggraph(layout) +
  geom_vline(xintercept = c(0, 2, 4), linetype = "dashed", colour = "grey80", linewidth = 0.3) +
  geom_edge_diagonal(
    aes(width = abs_weight, colour = sign),
    arrow     = arrow(length = unit(2, "mm"), type = "closed"),
    start_cap = circle(5, "mm"),
    end_cap   = circle(5, "mm")
  ) +
  geom_node_point(aes(colour = group), size = 12, shape = 19) +
  geom_node_text(        
    aes(label = name),         
    colour = palette$teal,         
    family = "space_grotesk",         
    size   = 3,        
    fontface = "bold"    
  ) +
  scale_colour_manual(
    values = c("input" = "grey20", "hidden" = "grey20", "output" = "grey20"),
    guide  = "none"
  ) +
  annotate("text", x = 0,  y = 4.5, label = "Input layer",  fontface = "bold", size = 4, family = "space_grotesk") +
  annotate("text", x = 2,  y = 4.5, label = "Hidden layer", fontface = "bold", size = 4, family = "space_grotesk") +
  annotate("text", x = 4,  y = 4.5, label = "Output layer", fontface = "bold", size = 4, family = "space_grotesk") +
  scale_edge_width(range = c(0.3, 3), guide = "none") +
  scale_edge_colour_manual(
    values = c("+ve" = "#14b8a6", "-ve" = "#ec4899"),
    name   = "Weight sign"
  ) +
  scale_fill_manual(
    values = c("input" = "#93c5fd", "hidden" = "#fde68a", "output" = "#fca5a5"),
    name   = "Layer",
    guide  = "none"
  ) +
  theme_graph() +
  theme(
    plot.title      = element_text(hjust = 0.5, family = "space_grotesk"),
    plot.subtitle   = element_text(hjust = 0.5, family = "space_grotesk"),
    plot.caption    = element_text(hjust = 0.5, family = "space_grotesk"),
    legend.text     = element_text(family = "space_grotesk"),
    legend.title    = element_text(family = "space_grotesk"),
    legend.position = 'bottom'
  ) +
  coord_cartesian(xlim = c(-0.2, 4.2), ylim = c(-3.5, 4.5)) +
  labs(
    title = "Trained neural network weights",
    caption = "Edge thickness reflects weight magnitude, colour reflects sign"
  )

ggsave(  
  file.path('plots', "nn-weights.jpg"),  
  device = 'jpg',  
  width = plot_dim$width,  
  height = plot_dim$height * 1.2,  
  units = 'px',  
  dpi = 250
)
