def stack_to_one_hot_df(stack: np.ndarray, list_id: list[str]) -> pd.DataFrame:
    """ """

    couples = ["_".join(couple) for couple in combinations(list_id, 2)]
    df = pd.DataFrame(columns=couples)

    for c in couples:
        idx = (list_id.index(c.split("_")[0]), list_id.index(c.split("_")[1]))

        df[c] = stack[:, *idx]

    return df.astype(bool)


def apriori_(df: pd.DataFrame, min_support: float, min_number: int) -> pd.DataFrame:
    motifs = apriori(df, min_support=min_support, use_colnames=True, low_memory=True)
    filtered_motifs = motifs[motifs["itemsets"].apply(len) >= min_number].reset_index()

    filtered_motifs.sort_values(by="support", ascending=False, inplace=True)

    return filtered_motifs


def apply_sousgraph_connexe_maximum(x: frozenset) -> frozenset:
    """
    renvoie les sommets de la plus grande composante connexe du motif

    """

    graph = nx.Graph()

    for edge in x:
        n1, n2 = edge.split("_")
        graph.add_edge(n1, n2)
    connected_components = nx.connected_components(graph)
    largest_component_nodes = max(connected_components, key=len)

    return frozenset(largest_component_nodes)


def get_maximum_connex_graph(dataframe: pd.DataFrame) -> pd.DataFrame:
    """ """

    dataframe["motif_connexe_maximum"] = dataframe["itemsets"].apply(
        apply_sousgraph_connexe_maximum
    )
    dataframe["longueur_graph_connex"] = dataframe["motif_connexe_maximum"].apply(len)

    dataframe.sort_values(by="longueur_graph_connex", inplace=True, ascending=False)

    return dataframe


def transform_table_to_SPADE(dataframe, C=60):
    """
    On fait une fenetre de C ligne ( exmple : lissage 60 s, 1h de fenetre --< c=60)


    """
    df = pd.DataFrame({"seq_id": [], "sequence": []})
    id_max = int(0)
    c = 0
    l = list()
    for index, row in dataframe.iterrows():
        if index % C == 0:
            if id_max != 0:
                df = pd.concat(
                    [df, pd.DataFrame({"seq_id": [id_max], "sequence": [l]})]
                )
                l = [row[row == True].index]
                id_max += 1

            else:
                l = [row[row == True].index]
                id_max += 1

        else:
            l.append(list(row[row == True].index))

    return df


def transform_df_to_txt(df, list_id):
    dict_id = dict()
    c = 1
    for i in list_id:
        for j in list_id:
            dict_id["{}_{}".format(i, j)] = c
            c += 1
    return dict_id

    txt = str()
    for index, row in df.iterrows():
        l = row["sequence"]
        for itemset in l:
            for item in itemset:
                txt += str(dict_id[item]) + " "
            txt += "-1 "

        txt += "\n"
    with open(
        "/Users/bouchet/Documents/Cours/Cours /AgroParisTech /3A/IODAA/PFR/TRACE_IODAA/savings/Output.txt",
        "w",
    ) as text_file:
        text_file.write(txt)
