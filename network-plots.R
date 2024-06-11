
library(dplyr)
library(igraph)

# SET WORKING DIRECTORY
WD <- '/data'

plotNetwork <- function(g, graph_layout, text_title){

  edge_width <- E(g)$weight / 8
  vertex_size <- degree(g) + 4

  plot(g,
       layout=graph_layout,
       vertex.size = vertex_size,
       edge.color = 'black',
       edge.width = edge_width,

       vertex.frame.width = 0.7,
       vertex.frame.color= 'black',
       vertex.label = NA,
       vertex.label.cex = 0.5,
       vertex.label.dist=1,
       vertex.label.font = NA,
       vertex.color = 'white'
  )
  title(text_title, cex.main = 1.5)

}

# LOAD NETWORKS
adj_G3 <- as.matrix(read.csv(paste0(WD, 'adj_G3.csv')))
g_G3 <- graph_from_adjacency_matrix(adj_G3, weighted = T, mode = 'plus')

adj_G4 <- as.matrix(read.csv(paste0(WD, 'adj_G4.csv')))
g_G4 <- graph_from_adjacency_matrix(adj_G4, weighted = T, mode = 'plus')

adj_SH <- as.matrix(read.csv(paste0(WD, 'adj_SH.csv')))
g_SH <- graph_from_adjacency_matrix(adj_SH, weighted = T, mode = 'plus')

adj_AE <- as.matrix(read.csv(paste0(WD, 'adj_AE.csv')))
g_AE <- graph_from_adjacency_matrix(adj_AE, weighted = T, mode = 'plus')

# PLOT NETWORKS
set.seed(123)
graph_layout <- layout.davidson.harel(g_AE)

svg("networks.svg")

par(mfrow=c(2,2),
    oma = c(0,0,0,0) + 0.1,
    mar = c(0,0,1,1) + 0.1)
plotNetwork(g_AE, graph_layout, 'Coder 1')
plotNetwork(g_SH, graph_layout, 'Coder 2')
plotNetwork(g_G3, graph_layout, 'GPT-3.5')
plotNetwork(g_G4, graph_layout, 'GPT-4')

dev.off()
