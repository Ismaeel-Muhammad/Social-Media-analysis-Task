import tkinter as tk
import random as random
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tkinter import filedialog
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from networkx.algorithms.community import greedy_modularity_communities
import tkinter.colorchooser
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
from networkx.algorithms.community.quality import modularity

# Define layout_type globally
layout_type = None
G = None
comparison_done = False

def select_nodes_file():
    global nodes_file_path
    nodes_file_path = filedialog.askopenfilename(initialdir="/", title="Select Nodes File",
                                                 filetypes=(("CSV files", ".csv"), ("all files", ".*")))
    nodes_label.config(text="Nodes File: " + nodes_file_path)
def select_edges_file():
    global edges_file_path
    edges_file_path = filedialog.askopenfilename(initialdir="/", title="Select Edges File",
                                                 filetypes=(("CSV files", ".csv"), ("all files", ".*")))
    edges_label.config(text="Edges File: " + edges_file_path)
def select_graph_type(event):
    global graph_type
    graph_type = graph_type_var.get()
def select_layout_type(event):
    global layout_type
    layout_type = layout_type_var.get()
def analyze_graph():
    global nodes_file_path, edges_file_path, graph_type, layout_type, G

    # read nodes and edges from CSV files
    nodes = pd.read_csv(nodes_file_path)
    edges = pd.read_csv(edges_file_path)

    # Create a graph
    if graph_type == "Undirected":
        G = nx.Graph()
    else:
        G = nx.DiGraph()

    # Add nodes and edges to the graph
    for index, row in nodes.iterrows():
        G.add_node(row['ID'], **row[1:].to_dict())  # Adding custom node attributes
    for index, row in edges.iterrows():
        G.add_edge(row['Source'], row['Target'], **row[2:].to_dict())  # Adding custom edge attributes

    # Analyze graph metrics
    clustering_coef = nx.average_clustering(G)
    if graph_type.__eq__("Undirected"):
        avg_path_length = nx.average_shortest_path_length(G)
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    avg_degree = sum(dict(G.degree()).values()) / num_nodes
    density = nx.density(G)
    node_degrees = dict(G.degree())

    # Create a new window to display the analysis results
    analysis_window = tk.Toplevel(root)
    analysis_window.title("Graph Analysis Results")

    # Format and display results
    result_text = f"Graph Analysis Summary:\n"
    result_text += f"-----------------------\n"
    result_text += f"Number of Nodes: {num_nodes}\n"
    result_text += f"Number of Edges: {num_edges}\n"
    result_text += f"Graph Density: {density:.4f}\n"
    if graph_type.__eq__("Undirected"):
        result_text += f"Average Shortest Path Length: {avg_path_length:.4f}\n"
    result_text += f"Average Clustering Coefficient: {clustering_coef:.4f}\n"
    result_text += "Average degree: {:.2f}\n".format(avg_degree)

    # Display results in a label
    result_label = ttk.Label(analysis_window, text=result_text, wraplength=400, justify="left")
    result_label.pack(padx=10, pady=10)

    # Add highlighted message
    highlight_label = ttk.Label(analysis_window, text="The Network is now initialized!", foreground="blue",
                                font=("Helvetica", 12, "bold"))
    highlight_label.pack(pady=10)
def visualize_graph():
    global node_attributes, edge_attributes

    if G is None:
        print("Please analyze the graph first.")
        return

    # Retrieve node and edge attributes from user input
    node_size = int(node_size_entry.get())
    node_color = node_color_entry.get()
    node_shape = node_shape_var.get()

    edge_color = edge_color_entry.get()
    edge_width = int(edge_width_entry.get())

    pos = nx.random_layout(G)
    if layout_type == "Spring":
        pos = nx.spring_layout(G)
    elif layout_type == "Kamada-Kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout_type == "Fruchterman-Reingold":
        pos = nx.fruchterman_reingold_layout(G)
    elif layout_type == "Spectral":
        pos = nx.spectral_layout(G)
    elif layout_type == "Shell":
        pos = nx.shell_layout(G)
    elif layout_type == "Planar":
        pos = nx.planar_layout(G)
    elif layout_type == "Spiral":
        pos = nx.spiral_layout(G)
    elif layout_type == "Circular":
        pos = nx.circular_layout(G)
    elif layout_type == "Rescale":
        pos = nx.rescale_layout(G)
    elif layout_type == "Multi-Partite":
        pos = nx.multipartite_layout(G)

    plt.figure(figsize=(8, 6))

    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color, node_shape=node_shape)
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, width=edge_width)

    plt.title("Graph Visualization")
    plt.axis("off")
    plt.show()
def visualize_graph_pro(graph):
    global node_attributes, edge_attributes

    if graph is None:
        print("Please provide a valid graph.")
        return

    # Retrieve node and edge attributes from user input
    node_size = int(node_size_entry.get())
    node_color = node_color_entry.get()
    node_shape = node_shape_var.get()

    edge_color = edge_color_entry.get()
    edge_width = int(edge_width_entry.get())

    pos = nx.random_layout(graph)
    if layout_type == "Spring":
        pos = nx.spring_layout(graph)
    elif layout_type == "Kamada-Kawai":
        pos = nx.kamada_kawai_layout(graph)
    elif layout_type == "Fruchterman-Reingold":
        pos = nx.fruchterman_reingold_layout(graph)
    elif layout_type == "Spectral":
        pos = nx.spectral_layout(graph)
    elif layout_type == "Shell":
        pos = nx.shell_layout(graph)
    elif layout_type == "Planar":
        pos = nx.planar_layout(graph)
    elif layout_type == "Spiral":
        pos = nx.spiral_layout(graph)
    elif layout_type == "Circular":
        pos = nx.circular_layout(graph)
    elif layout_type == "Rescale":
        pos = nx.rescale_layout(graph)
    elif layout_type == "Multi-Partite":
        pos = nx.multipartite_layout(graph)

    plt.figure(figsize=(8, 6))

    # Draw nodes and edges
    nx.draw_networkx_nodes(graph, pos, node_size=node_size, node_color=node_color, node_shape=node_shape)
    nx.draw_networkx_edges(graph, pos, edge_color=edge_color, width=edge_width)

    plt.title("Graph Visualization")
    plt.axis("off")
    plt.show()
def filter_nodes_by_centrality():

    choice_index = centrality_options.index(centrality_var.get())
    if choice_index == 1:
        degree()
    elif choice_index == 2:
        closeness()
    elif choice_index == 3:
        betweenness()
    else:
        print("Error")
def degree():
    filtered_nodes = []
    filtered_edges = []
    centrality_values = None
    threshold = float(threshold_entry.get())
    centrality_values = nx.degree_centrality(G)
    for node, centrality in centrality_values.items():
        if centrality >= threshold:
            filtered_nodes.append(node)
    if len(filtered_nodes) == 0:
        tk.messagebox.showerror("Error", "This Degree doesn't exists.")
    else:
        filtered_edges = get_filtered_edges(G, filtered_nodes)
        if graph_type == "Undirected":
            filtered_network = nx.Graph()
        else:
            filtered_network = nx.DiGraph()
        filtered_network.add_nodes_from(filtered_nodes)
        filtered_network.add_edges_from(filtered_edges)
        visualize_graph_pro(filtered_network)
def betweenness():
    filtered_nodes = []
    filtered_edges = []
    centrality_values = None
    threshold = float(threshold_entry.get())
    centrality_values = nx.betweenness_centrality(G)
    for node, centrality in centrality_values.items():
        if centrality >= threshold:
            filtered_nodes.append(node)
    if len(filtered_nodes) == 0:
        tk.messagebox.showerror("Error", "This Betweenness doesn't exists.")
    else:
        filtered_edges = get_filtered_edges(G, filtered_nodes)
        if graph_type == "Undirected":
            filtered_network = nx.Graph()
        else:
            filtered_network = nx.DiGraph()
        filtered_network.add_nodes_from(filtered_nodes)
        filtered_network.add_edges_from(filtered_edges)
        visualize_graph_pro(filtered_network)
def closeness():
    filtered_nodes = []
    filtered_edges = []
    centrality_values = None
    threshold = float(threshold_entry.get())
    centrality_values = nx.closeness_centrality(G)
    for node, centrality in centrality_values.items():
        if centrality >= threshold:
            filtered_nodes.append(node)
    if len(filtered_nodes) == 0:
        tk.messagebox.showerror("Error", "This Closeness doesn't exists.")
    else:
        filtered_edges = get_filtered_edges(G, filtered_nodes)
        if graph_type == "Undirected":
            filtered_network = nx.Graph()
        else:
            filtered_network = nx.DiGraph()
        filtered_network.add_nodes_from(filtered_nodes)
        filtered_network.add_edges_from(filtered_edges)
        visualize_graph_pro(filtered_network)
def filter_nodes_by_centrality_interval():

    choice_index = centrality_options.index(centrality_var.get())
    if choice_index == 1:
        degree_range()
    elif choice_index == 2:
        closeness_range()
    elif choice_index == 3:
        betweenness_range()
    else:
        print("Error")
def degree_range():
    filtered_nodes = []
    filtered_edges = []
    centrality_values = None
    threshold = float(from_entry.get())
    threshold1 = float(to_entry.get())
    centrality_values = nx.degree_centrality(G)
    for node, centrality in centrality_values.items():
        if threshold <= centrality <= threshold1:
            filtered_nodes.append(node)
    if len(filtered_nodes) == 0:
        tk.messagebox.showerror("Error", "This Degree Range doesn't exists.")
    else:
        filtered_edges = get_filtered_edges(G, filtered_nodes)
        if graph_type == "Undirected":
            filtered_network = nx.Graph()
        else:
            filtered_network = nx.DiGraph()
        filtered_network.add_nodes_from(filtered_nodes)
        filtered_network.add_edges_from(filtered_edges)
        visualize_graph_pro(filtered_network)
def betweenness_range():
    filtered_nodes = []
    filtered_edges = []
    centrality_values = None
    threshold = float(from_entry.get())
    threshold1 = float(to_entry.get())
    centrality_values = nx.betweenness_centrality(G)
    for node, centrality in centrality_values.items():
        if threshold <= centrality <= threshold1:
            filtered_nodes.append(node)
    if len(filtered_nodes) == 0:
        tk.messagebox.showerror("Error", "This Betweenness Range doesn't exists.")
    else:
        filtered_edges = get_filtered_edges(G, filtered_nodes)
        if graph_type == "Undirected":
            filtered_network = nx.Graph()
        else:
            filtered_network = nx.DiGraph()
        filtered_network.add_nodes_from(filtered_nodes)
        filtered_network.add_edges_from(filtered_edges)
        visualize_graph_pro(filtered_network)
def closeness_range():
    filtered_nodes = []
    filtered_edges = []
    centrality_values = None
    threshold = float(from_entry.get())
    threshold1 = float(to_entry.get())
    centrality_values = nx.closeness_centrality(G)
    for node, centrality in centrality_values.items():
        if threshold <= centrality <= threshold1:
            filtered_nodes.append(node)
    if len(filtered_nodes) == 0:
        tk.messagebox.showerror("Error", "This Closeness Range doesn't exists.")
    else:
        filtered_edges = get_filtered_edges(G, filtered_nodes)
        if graph_type == "Undirected":
            filtered_network = nx.Graph()
        else:
            filtered_network = nx.DiGraph()
        filtered_network.add_nodes_from(filtered_nodes)
        filtered_network.add_edges_from(filtered_edges)
        visualize_graph_pro(filtered_network)
def get_filtered_edges(graph, filtered_nodes):
    filtered_edges = [(u, v) for u, v in graph.edges() if u in filtered_nodes and v in filtered_nodes]
    return filtered_edges
# Function to filter nodes based on community membership
def filter_nodes_by_community(G, community):
    communities = nx.algorithms.community.greedy_modularity_communities(G)
    for idx, comm in enumerate(communities):
        if community in comm:
            return G.subgraph(comm)
    return nx.Graph()
def choices():
    global community_algorithm , comparison_done
    if not comparison_done:
        print("Please Compare the Community Detection Algorithms outputs first")
        return
    print("pressd")
    gn_communities, communities_louvain = compare_community_detection()
    if community_algorithm.get() == "Girvan-Newman":
        print(f"Graphing {community_algorithm.get()} graph")
        visualize_communities("Girvan-Newman", gn_communities)
    elif community_algorithm.get() == "Louvain":
        print(f"Graphing {community_algorithm.get()} graph")
        visualize_communities("Louvain", communities_louvain )
    else:
        print("Error selecting")
def visualize_communities(community_algorithm, communities):
    if G is None:
        print("Please analyze the graph first.")
        return
    # Create a plot figure
    plt.figure(figsize=(8, 6))
    # Draw the graph with selected layout
    pos = nx.random_layout(G)
    if layout_type == "Spring":
        pos = nx.spring_layout(G)
    elif layout_type == "Kamada-Kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout_type == "Fruchterman-Reingold":
        pos = nx.fruchterman_reingold_layout(G)
    elif layout_type == "Spectral":
        pos = nx.spectral_layout(G)
    elif layout_type == "Shell":
        pos = nx.shell_layout(G)
    elif layout_type == "Planar":
        pos = nx.planar_layout(G)
    elif layout_type == "Spiral":
        pos = nx.spiral_layout(G)
    elif layout_type == "Circular":
        pos = nx.circular_layout(G)
    elif layout_type == "Rescale":
        pos = nx.rescale_layout(G)
    elif layout_type == "Multi-Partite":
        pos = nx.multipartite_layout(G)

    partition = {node: cid for cid, Community in enumerate(communities) for node in Community}
    community_colors = {}
    for node, community_id in partition.items():
        if community_id not in community_colors:
            community_colors[community_id] = (random.random(), random.random(), random.random())
        nx.set_node_attributes(G, {node: {'community': community_id}})

    node_colors = [community_colors[partition[node]] for node in G.nodes]

    # Assign colors to nodes based on community membership
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size= 60)
    nx.draw_networkx_edges(G, pos, alpha=0.5)

    # Add node labels
    nx.draw_networkx_labels(G, pos)

    # Display the plot
    plt.title(f"{community_algorithm} Visualization")
    plt.show()
def compare_community_detection():
    global comparison_done, gn_communities, communities_louvain
    if comparison_done:
        num_gn_communities = len(gn_communities)
        num_louvain_communities = len(communities_louvain)
        #Create a new window to display the results
        # results_window = tk.Toplevel()
        # results_window.title("Community Detection Comparison")

        # label_num_gn_communities = ttk.Label(results_window,
        #                                      text=f"Number of Girvan-Newman Communities: {num_gn_communities}   \n")
        # label_num_gn_communities.pack()
        #
        # label_num_louvain_communities = ttk.Label(results_window,
        #                                           text=f"Number of Louvain Communities: {num_louvain_communities}   \n")
        # label_num_louvain_communities.pack()
        # print("Girvan-Newman Communities:", gn_communities)
        # print("Louvain Communities:", communities_louvain)

        print("\nNumber of Girvan-Newman Communities:", num_gn_communities)
        print("Number of Louvain Communities:", num_louvain_communities)
        #visualize_communities("Girvan-Newman", gn_communities)
        #visualize_communities("Louvain", communities_louvain)
    else:
        if G is None:
            print("Please analyze the graph first.")
            return
        communities_gn = None
        communities_louvain = None
        # Apply community detection algorithms
        communities_gn = nx.algorithms.community.girvan_newman(G)
        print("Hot Damn")
        communities_louvain = nx.algorithms.community.greedy_modularity_communities(G)
        print("Hot DAMN")
        gn_communities = next(communities_gn)
        print("HOT DAMN")
        comparison_done = True


        # Create a new window to display the results
        results_window = tk.Toplevel()
        results_window.title("Community Detection Comparison")

        num_gn_communities = len(gn_communities)
        num_louvain_communities = len(communities_louvain)

        label_num_gn_communities = ttk.Label(results_window,
                                             text=f"Number of Girvan-Newman Communities: {num_gn_communities}   \n")
        label_num_gn_communities.pack()

        label_num_louvain_communities = ttk.Label(results_window,
                                                  text=f"Number of Louvain Communities: {num_louvain_communities}   \n")
        label_num_louvain_communities.pack()
        print("Girvan-Newman Communities:", gn_communities)
        print("Louvain Communities:", communities_louvain)

        print("\nNumber of Girvan-Newman Communities:", num_gn_communities)
        print("Number of Louvain Communities:", num_louvain_communities)
        visualize_communities("Girvan-Newman", gn_communities)
        visualize_communities("Louvain", communities_louvain)
        print("Finished comparing")
    print(type(gn_communities),"\t",type(communities_louvain))
    return gn_communities, communities_louvain
def select_and_visualize_community():
    if G is None:
        print("Please analyze the graph first.")
        return
    if not comparison_done:
        print("Please Compare the Community Detection Algorithms outputs first")
        return
    def visualize_selected():
        community_index = int(community_index_entry.get())
        visualize_selected_community(community_index, community_algorithm)
    # Create a new window to select community to visualize
    community_window = tk.Toplevel()
    community_window.title("Select Community to Visualize")
    community_index_label = ttk.Label(community_window, text="Enter Community Index:")
    community_index_label.pack()
    community_index_entry = ttk.Entry(community_window)
    community_index_entry.pack()
    visualizing_button= ttk.Button(community_window, text="Visualize", command=visualize_selected)
    visualizing_button.pack()
def tuple_of_sets_to_graphs(tuple_of_sets, original_graph):
    graphs = []
    for node_set in tuple_of_sets:
        new_graph = nx.Graph()  # Or use DiGraph() if your original graph is directed
        new_graph.add_nodes_from(node_set)
        for node1 in node_set:
            for node2 in node_set:
                if original_graph.has_edge(node1, node2):
                    new_graph.add_edge(node1, node2)
        graphs.append(new_graph)
    return graphs
def frozensets_to_graphs(list_of_frozensets, original_graph):
    graphs = []
    for frozen_set_of_nodes in list_of_frozensets:
        # Create a new graph object
        new_graph = nx.Graph()  # Or use DiGraph() if your original graph is directed

        # Add nodes from the frozenset to the new graph
        new_graph.add_nodes_from(frozen_set_of_nodes)

        # Add edges between nodes in the frozenset from the original graph
        for node1 in frozen_set_of_nodes:
            for node2 in frozen_set_of_nodes:
                if original_graph.has_edge(node1, node2):
                    new_graph.add_edge(node1, node2)

        graphs.append(new_graph)

    return graphs
def visualize_selected_community(community_index, community_algorithm):
    gn_communities, communities_louvain = compare_community_detection()
    detected_communities = None
    communities_list = None
    # Retrieve node and edge attributes from user input
    node_size = int(node_size_entry.get())
    node_color = node_color_entry.get()
    node_shape = node_shape_var.get()
    edge_color = edge_color_entry.get()
    edge_width = int(edge_width_entry.get())

    if gn_communities is not None or communities_louvain is not None:
        if community_algorithm.get() == "Girvan-Newman":
            detected_communities = tuple_of_sets_to_graphs(gn_communities,G)
        elif community_algorithm.get() == "Louvain":
            detected_communities = frozensets_to_graphs(communities_louvain,G)
        selected_community = detected_communities[community_index]
        if selected_community:
            plt.figure(figsize=(8, 6))
            # Draw the graph with selected layout
            pos = nx.random_layout(selected_community)
            if layout_type == "Spring":
                pos = nx.spring_layout(selected_community)
            elif layout_type == "Kamada-Kawai":
                pos = nx.kamada_kawai_layout(selected_community)
            elif layout_type == "Fruchterman-Reingold":
                pos = nx.fruchterman_reingold_layout(selected_community)
            elif layout_type == "Spectral":
                pos = nx.spectral_layout(selected_community)
            elif layout_type == "Shell":
                pos = nx.shell_layout(selected_community)
            elif layout_type == "Planar":
                pos = nx.planar_layout(selected_community)
            elif layout_type == "Spiral":
                pos = nx.spiral_layout(selected_community)
            elif layout_type == "Circular":
                pos = nx.circular_layout(selected_community)
            elif layout_type == "Rescale":
                pos = nx.rescale_layout(selected_community)
            elif layout_type == "Multi-Partite":
                pos = nx.multipartite_layout(selected_community)

            # Assign colors to nodes based on community membership
            nx.draw_networkx_nodes(selected_community, pos, node_color=node_color, node_size=node_size,node_shape=node_shape)
            nx.draw_networkx_edges(selected_community, pos, alpha=0.5, width=edge_width,edge_color=edge_color)

            # Add node labels
            nx.draw_networkx_labels(selected_community, pos)

            # Display the plot
            plt.title(f"The Community #{community_index} Detected By The {community_algorithm} Algorithm")
            plt.show()
    else:
        print("Why is this happening to me")
def evaluate_clustering0():
    evaluate_clustering(clustering_method_var.get())
def evaluate_clustering(method):

    if G is None:
        print("Please analyze the graph first.")
        return

    global partitioned_communities, predicted_labels ,community
    partitioned_communities = None
    if method == "Kernighan-Lin":
        partitioned_communities = list(nx.algorithms.community.greedy_modularity_communities(G))
        best_modularity = modularity(G, partitioned_communities)
        best_partition = partitioned_communities

        for i in range(len(partitioned_communities)):
            for j in range(i + 1, len(partitioned_communities)):
                community1 = partitioned_communities[i]
                community2 = partitioned_communities[j]

                # Compute modularity change after swapping nodes between communities
                for node1 in community1:
                    for node2 in community2:
                        new_community1 = community1 - {node1} | {node2}
                        new_community2 = community2 - {node2} | {node1}
                        new_partition = partitioned_communities.copy()
                        new_partition[i] = new_community1
                        new_partition[j] = new_community2
                        new_modularity = modularity(G, new_partition)

                        # Update if modularity improves
                        if new_modularity > best_modularity:
                            best_modularity = new_modularity
                            best_partition = new_partition

        partitioned_communities = best_partition
        predicted_labels = [label for label, community in enumerate(partitioned_communities) for _ in community]

    elif method == "Spectral Clustering":
        partitioned_communities = list(nx.algorithms.community.greedy_modularity_communities(G))
        predicted_labels = SpectralClustering(n_clusters=len(partitioned_communities), affinity='nearest_neighbors', assign_labels='kmeans').fit_predict(nx.to_numpy_array(G))

    elif method == "Greedy Modularity":
        # Use Greedy Modularity from the community library
        partitioned_communities = list(greedy_modularity_communities(G))
        predicted_labels = [label for label, community in enumerate(partitioned_communities) for _ in
                            range(len(community))]

    if partitioned_communities:
        true_labels = [label for label, community in enumerate(partitioned_communities) for _ in community]

        # Calculate silhouette score
        silhouette = silhouette_score(nx.to_numpy_array(G), predicted_labels)

        # Calculate NMI
        nmi = normalized_mutual_info_score(true_labels, predicted_labels)

        # Display results in a messagebox
        result_message = f"Clustering Evaluation Result Using {method} for Clustering:\n\nModularity: {nx.algorithms.community.modularity(G, partitioned_communities)}\nSilhouette Score: {silhouette}\nNormalized Mutual Information (NMI): {nmi}"
        tk.messagebox.showinfo("Clustering Evaluation", result_message)
    else:
        print("Please Choose a method")
def link_analysis():
    # Assuming G is a predefined global variable or any other graph available in the scope
    if G is None:
        print("Please analyze the graph first.")
        return

    # Calculate PageRank
    pagerank_scores = nx.pagerank(G)

    # Calculate Betweenness Centrality
    betweenness_centrality = nx.betweenness_centrality(G)

    # Calculate Closeness Centrality
    closeness_centrality = nx.closeness_centrality(G)

    # Calculate Degree Centrality
    degree_centrality = nx.degree_centrality(G)

    # Calculate summary statistics
    max_pagerank_node = max(pagerank_scores, key=pagerank_scores.get)
    min_pagerank_node = min(pagerank_scores, key=pagerank_scores.get)
    avg_pagerank = np.mean(list(pagerank_scores.values()))

    max_betweenness_node = max(betweenness_centrality, key=betweenness_centrality.get)
    min_betweenness_node = min(betweenness_centrality, key=betweenness_centrality.get)
    avg_betweenness = np.mean(list(betweenness_centrality.values()))

    max_closeness_node = max(closeness_centrality, key=closeness_centrality.get)
    min_closeness_node = min(closeness_centrality, key=closeness_centrality.get)
    avg_closeness = np.mean(list(closeness_centrality.values()))

    max_degree_node = max(degree_centrality, key=degree_centrality.get)
    min_degree_node = min(degree_centrality, key=degree_centrality.get)
    avg_degree = np.mean(list(degree_centrality.values()))

    # Create a GUI window
    analysis_win = tk.Tk()
    analysis_win.title("Link Analysis Results")

    # Create a frame for displaying results
    result_frame = ttk.Frame(analysis_win)
    result_frame.pack(padx=10, pady=10)

    # Display Summary Labels
    ttk.Label(result_frame, text="Summary:", font=('Helvetica', 14, 'bold')).pack(pady=5)

    # PageRank Summary
    ttk.Label(result_frame, text="PageRank Summary:", font=('Helvetica', 12, 'bold')).pack(anchor="w")
    ttk.Label(result_frame, text=f"Max Node: {max_pagerank_node}, Max Score: {pagerank_scores[max_pagerank_node]:.4f}", font=('Helvetica', 10)).pack(anchor="w")
    ttk.Label(result_frame, text=f"Min Node: {min_pagerank_node}, Min Score: {pagerank_scores[min_pagerank_node]:.4f}", font=('Helvetica', 10)).pack(anchor="w")
    ttk.Label(result_frame, text=f"Avg Score: {avg_pagerank:.4f}", font=('Helvetica', 10)).pack(anchor="w")

    # Betweenness Centrality Summary
    ttk.Label(result_frame, text="Betweenness Centrality Summary:", font=('Helvetica', 12, 'bold')).pack(anchor="w")
    ttk.Label(result_frame, text=f"Max Node: {max_betweenness_node}, Max Centrality: {betweenness_centrality[max_betweenness_node]:.4f}", font=('Helvetica', 10)).pack(anchor="w")
    ttk.Label(result_frame, text=f"Min Node: {min_betweenness_node}, Min Centrality: {betweenness_centrality[min_betweenness_node]:.4f}", font=('Helvetica', 10)).pack(anchor="w")
    ttk.Label(result_frame, text=f"Avg Centrality: {avg_betweenness:.4f}", font=('Helvetica', 10)).pack(anchor="w")

    # Closeness Centrality Summary
    ttk.Label(result_frame, text="Closeness Centrality Summary:", font=('Helvetica', 12, 'bold')).pack(anchor="w")
    ttk.Label(result_frame, text=f"Max Node: {max_closeness_node}, Max Centrality: {closeness_centrality[max_closeness_node]:.4f}", font=('Helvetica', 10)).pack(anchor="w")
    ttk.Label(result_frame, text=f"Min Node: {min_closeness_node}, Min Centrality: {closeness_centrality[min_closeness_node]:.4f}", font=('Helvetica', 10)).pack(anchor="w")
    ttk.Label(result_frame, text=f"Avg Centrality: {avg_closeness:.4f}", font=('Helvetica', 10)).pack(anchor="w")

    # Degree Centrality Summary
    ttk.Label(result_frame, text="Degree Centrality Summary:", font=('Helvetica', 12, 'bold')).pack(anchor="w")
    ttk.Label(result_frame, text=f"Max Node: {max_degree_node}, Max Centrality: {degree_centrality[max_degree_node]:.4f}", font=('Helvetica', 10)).pack(anchor="w")
    ttk.Label(result_frame, text=f"Min Node: {min_degree_node}, Min Centrality: {degree_centrality[min_degree_node]:.4f}", font=('Helvetica', 10)).pack(anchor="w")
    ttk.Label(result_frame, text=f"Avg Centrality: {avg_degree:.4f}", font=('Helvetica', 10)).pack(anchor="w")

    # Plot histograms
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))

    axs[0, 0].hist(list(pagerank_scores.values()), bins=20, color='blue', alpha=0.7)
    axs[0, 0].set_title('PageRank Scores')
    axs[0, 0].set_xlabel('Score')
    axs[0, 0].set_ylabel('Frequency')

    axs[0, 1].hist(list(betweenness_centrality.values()), bins=20, color='green', alpha=0.7)
    axs[0, 1].set_title('Betweenness Centrality')
    axs[0, 1].set_xlabel('Centrality')
    axs[0, 1].set_ylabel('Frequency')

    axs[1, 0].hist(list(closeness_centrality.values()), bins=20, color='red', alpha=0.7)
    axs[1, 0].set_title('Closeness Centrality')
    axs[1, 0].set_xlabel('Centrality')
    axs[1, 0].set_ylabel('Frequency')

    axs[1, 1].hist(list(degree_centrality.values()), bins=20, color='purple', alpha=0.7)
    axs[1, 1].set_title('Degree Centrality')
    axs[1, 1].set_xlabel('Centrality')
    axs[1, 1].set_ylabel('Frequency')

    #plt.tight_layout()

    # Convert matplotlib figure to tkinter compatible canvas
    canvas = FigureCanvasTkAgg(fig, master=analysis_win)
    #canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    analysis_win.mainloop()
def pick_node_color():
    color = tkinter.colorchooser.askcolor()[1]
    if color:
        node_color_entry.delete(0, tk.END)
        node_color_entry.insert(tk.END, color)
def pick_edge_color():
    color = tkinter.colorchooser.askcolor()[1]
    if color:
        edge_color_entry.delete(0, tk.END)
        edge_color_entry.insert(tk.END, color)
def select_node_shape(event):
    global shape_mapping
    selected_shape = event
    mapped_shape = shape_mapping[selected_shape]
    node_shape_var.set(mapped_shape)

root = tk.Tk()
root.geometry("700x900")
root.title("Mini Social Network Analysis Tool")

# Nodes section
nodes_frame = ttk.Frame(root, padding=(10, 10))
nodes_frame.pack(fill=tk.BOTH)

nodes_label = ttk.Label(nodes_frame, text="Nodes File: ")
nodes_label.grid(column=0, row=0, sticky=tk.W)

nodes_button = ttk.Button(nodes_frame, text="Select Nodes File", command=select_nodes_file)
nodes_button.grid(column=1, row=0, padx=10, pady=10)

# Edges section
edges_frame = ttk.Frame(root, padding=(10, 10))
edges_frame.pack(fill=tk.BOTH)

edges_label = ttk.Label(edges_frame, text="Edges File: ")
edges_label.grid(column=0, row=0, sticky=tk.W)

edges_button = ttk.Button(edges_frame, text="Select Edges File", command=select_edges_file)
edges_button.grid(column=1, row=0, padx=10, pady=5)

# GUI elements for node attributes
node_attributes_frame = ttk.Frame(root, padding=(10, 10))
node_attributes_frame.pack(fill=tk.BOTH)

node_size_label = ttk.Label(node_attributes_frame, text="Node Size:")
node_size_label.grid(column=0, row=0, sticky=tk.W)

node_size_entry = ttk.Entry(node_attributes_frame)
node_size_entry.grid(column=1, row=0)
node_size_entry.insert(tk.END, "100")

node_color_label = ttk.Label(node_attributes_frame, text="Node Color:")
node_color_label.grid(column=0, row=1, sticky=tk.W)

node_color_entry = ttk.Entry(node_attributes_frame)
node_color_entry.grid(column=1, row=1)
node_color_entry.insert(tk.END, "red")

node_color_button = ttk.Button(node_attributes_frame, text="Pick Color", command=pick_node_color)
node_color_button.grid(column=2, row=1)

node_shape_label = ttk.Label(node_attributes_frame, text="Node Shape:")
node_shape_label.grid(column=0, row=3, sticky=tk.W)

node_shape_options = ["Circle", "Square", "Triangle (up)", "Triangle (down)", "Triangle (right)", "Triangle (left)",
                      "Diamond", "Pentagon", "Hexagon", "Octagon"]
node_shape_var = tk.StringVar()
node_shape_var.set("o")
node_shape_optionmenu = ttk.OptionMenu(node_attributes_frame, node_shape_var, "o", *node_shape_options,
                                       command=select_node_shape)
node_shape_optionmenu.grid(column=1, row=3)

# Mapping from shape names to symbols
shape_mapping = {
    "Circle": "o",
    "Square": "s",
    "Triangle (up)": "^",
    "Triangle (down)": "v",
    "Triangle (right)": ">",
    "Triangle (left)": "<",
    "Diamond": "d",
    "Pentagon": "p",
    "Hexagon": "h",
    "Octagon": "8"
}

# GUI elements for edge attributes
edge_attributes_frame = ttk.Frame(root, padding=(10, 10))
edge_attributes_frame.pack(fill=tk.BOTH)

edge_color_label = ttk.Label(edge_attributes_frame, text="Edge Color:")
edge_color_label.grid(column=0, row=0, sticky=tk.W)

edge_color_entry = ttk.Entry(edge_attributes_frame)
edge_color_entry.grid(column=1, row=0)
edge_color_entry.insert(tk.END, "black")

edge_color_button = ttk.Button(edge_attributes_frame, text="Pick Color", command=pick_edge_color)
edge_color_button.grid(column=2, row=0)

edge_width_label = ttk.Label(edge_attributes_frame, text="Edge Width:")
edge_width_label.grid(column=0, row=1, sticky=tk.W)

edge_width_entry = ttk.Entry(edge_attributes_frame)
edge_width_entry.grid(column=1, row=1)
edge_width_entry.insert(tk.END, "1")

# Graph type section
graph_type_frame = ttk.Frame(root, padding=(10, 10))
graph_type_frame.pack(fill=tk.BOTH)

graph_type_label = ttk.Label(graph_type_frame, text="Graph Type: ")
graph_type_label.grid(column=0, row=0, sticky=tk.W)

graph_type_options = ["Choose type", "Directed", "Undirected"]
graph_type_var = tk.StringVar(root)
graph_type_var.set(graph_type_options[0])
graph_type_menu = ttk.OptionMenu(graph_type_frame, graph_type_var, *graph_type_options, command=select_graph_type)
graph_type_menu.grid(column=1, row=0, padx=10, pady=10)

# Layout type section
layout_type_frame = ttk.Frame(root, padding=(10, 10))
layout_type_frame.pack(fill=tk.BOTH)

layout_type_label = ttk.Label(layout_type_frame, text="Layout Type: ")
layout_type_label.grid(column=0, row=0, sticky=tk.W)

# Define the layout options for force-directed algorithms and hierarchical algorithms
force_directed_options = ["Spring", "Kamada-Kawai", "Fruchterman-Reingold", "Spectral", "Spiral"]
Specific_structured_options = ["Circular", "Shell", "Circos"]

layout_type_var = tk.StringVar(root)
layout_type_var.set(force_directed_options[0])  # Default to the first force-directed algorithm
layout_options = force_directed_options + Specific_structured_options

layout_type_menu = ttk.OptionMenu(layout_type_frame, layout_type_var, force_directed_options[0], *layout_options,
                                  command=select_layout_type)

# Add a disabled line between force-directed and hierarchical algorithms
layout_type_menu["menu"].insert_separator(len(force_directed_options))
layout_type_menu.grid(column=1, row=0, padx=10, pady=10)

# GUI elements for centrality selection
centrality_frame = ttk.Frame(root, padding=(10, 10))
centrality_frame.pack(fill=tk.BOTH)

centrality_label = ttk.Label(centrality_frame, text="Centrality Measure: ")
centrality_label.grid(column=0, row=0, sticky=tk.W)

centrality_options = ["Choose measurement", "Degree", "Closeness", "Betweenness"]
centrality_var = tk.StringVar(root)
centrality_var.set(centrality_options[0])

# Define dropdown menu for centrality selection
centrality_menu = ttk.OptionMenu(centrality_frame, centrality_var, *centrality_options)
centrality_menu.grid(column=1, row=0, padx=10, pady=10)

# Define a threshold value entry
threshold_label = ttk.Label(centrality_frame, text="Threshold:")
threshold_label.grid(column=0, row=1, sticky=tk.W)

threshold_entry = ttk.Entry(centrality_frame)
threshold_entry.grid(column=1, row=1, padx=10, pady=10)

# Button for applying filter
apply_filter_button = ttk.Button(centrality_frame, text="Apply Filter", command=filter_nodes_by_centrality,width=20)
apply_filter_button.grid(column=2, row=1, padx=5, pady=5)

# Label for the first textbox
from_label = ttk.Label(centrality_frame, text="From:")
from_label.grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)

# First textbox
from_entry = ttk.Entry(centrality_frame)
from_entry.grid(row=3, column=1, padx=5, pady=5)

# Label for the second textbox
to_label = ttk.Label(centrality_frame, text="To:")
to_label.grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)

# Second textbox
to_entry = ttk.Entry(centrality_frame)
to_entry.grid(row=4, column=1, padx=5, pady=5)

# Button for filtering
filter_button = ttk.Button(centrality_frame, text="Filter by Interval", command=filter_nodes_by_centrality_interval,width=20)
filter_button.grid(row=4, column=2, padx=5, pady=5)

# GUI elements for selecting the community detection algorithm
community_frame = ttk.Frame(root, padding=(10, 10))
community_frame.pack(fill=tk.BOTH)

community_label = ttk.Label(community_frame, text="Community Detection Algorithm: ")
community_label.grid(column=0, row=0, sticky=tk.W)

algorithm_options = ["Girvan-Newman", "Louvain"]
community_algorithm = tk.StringVar(root)
community_algorithm.set(algorithm_options[0])

# Define radio buttons for algorithm selection
for index, algorithm in enumerate(algorithm_options):
    ttk.Radiobutton(community_frame, text=algorithm, variable=community_algorithm, value=algorithm).grid(
        column=index + 1, row=0, padx=5, pady=5)

# Button to visualize communities
visualize_button = ttk.Button(community_frame, text="Visualize Communities", command=choices, width=20)
visualize_button.grid(column=3, row=0, padx=5, pady=5)

# GUI elements for selecting the clustering method
clustering_method_frame = ttk.Frame(root, padding=(10, 10))
clustering_method_frame.pack(fill=tk.BOTH)

clustering_method_label = ttk.Label(clustering_method_frame, text="Clustering Method: ")
clustering_method_label.grid(column=0, row=0, sticky=tk.W)

# Define the clustering method options
clustering_method_options = ["Choose Clustering Algorithm", "Kernighan-Lin", "Spectral Clustering", "Greedy Modularity"]
clustering_method_var = tk.StringVar(root)
clustering_method_var.set(clustering_method_options[0])

# Create dropdown menu for selecting clustering method
combo_method = ttk.OptionMenu(clustering_method_frame, clustering_method_var, *clustering_method_options)
combo_method.grid(column=1, row=0, padx=10, pady=10)

# Evaluate clustering button
evaluate_clustering_button = ttk.Button(clustering_method_frame, text="Evaluate Clustering",
                                        command=evaluate_clustering0, width=20)
evaluate_clustering_button.grid(column=2, row=0, padx=5, pady=5)

# Create a frame for the buttons
button_frame = ttk.Frame(root)
button_frame.pack()

# Create a style
style = ttk.Style()

# Configure the style to make the button green
style.configure("Green.TButton", foreground="blue", background="blue")

# Analyze graph button
analyze_graph_button = ttk.Button(button_frame, text="Analyze Graph", command=analyze_graph, width=20,
                                  style="Green.TButton")
analyze_graph_button.grid(row=0, column=0, padx=5, pady=5)

# Visualize graph button
visualize_graph_button = ttk.Button(button_frame, text="Visualize Graph", command=visualize_graph, width=20)
visualize_graph_button.grid(row=0, column=1, padx=5, pady=5)

# Compare community detection button
compare_community_detection_button = ttk.Button(button_frame, text="Community Detection",
                                                command=compare_community_detection, width=20)
compare_community_detection_button.grid(row=1, column=0, padx=5, pady=5)

# Partition graph button
partition_graph_button = ttk.Button(button_frame, text="Partition Graph", command=select_and_visualize_community, width=20)
partition_graph_button.grid(row=1, column=1, padx=5, pady=5)

# Evaluate clustering button
evaluate_clustering_button = ttk.Button(button_frame, text="Evaluate Clustering", command=evaluate_clustering0,
                                        width=20)
evaluate_clustering_button.grid(row=2, column=0, padx=5, pady=5)

# Link analysis button
link_analysis_button = ttk.Button(button_frame, text="Link Analysis", command=link_analysis, width=20)
link_analysis_button.grid(row=2, column=1, padx=5, pady=5)

root.mainloop()
