def sampling_function(G, alpha, k, partition_sets, xadj, adjncy, sample_count=15):
    for node in range(len(G.nodes)):
        neighbors_set = adjncy[xadj[node]:(xadj[node + 1])]
        print()
        print("Neighbors of node {} are".format(node), neighbors_set)

        node_in_partition_set = None

        for partition in partition_sets:
            if node in partition:
                node_in_partition_set = partition
                break

        # inter = node_in_partition_set.intersection(neighbors_set)
        inter = set(node_in_partition_set).intersection(neighbors_set)
        x = len(inter)
        diff = set(neighbors_set) - inter
        y = len(diff)

        print("Own partition neighbors of node {} is:-".format(node), inter)
        inter_avg.append(len(inter))
        print("Different partition neighbors of node {} is:-".format(node), diff)
        diff_avg.append(len(diff))

        w1 = []
        w2 = []
        for i in range(x):
            w1.append(alpha)

        for i in range(y):
            w2.append(1)

        sample_picks = []
        for i in range(sample_count):
            sample_pick = random.choices(population=list(inter) + list(diff), weights=list(w1) + list(w2), k=k)
            sample_picks.append(sample_pick)

            if node in output:
                output[node].append(sample_pick)
            else:
                output[node] = [sample_pick]

        flat_list = [item for sublist in sample_picks for item in sublist]
        print("neighbors node after sampling", flat_list)

        def intersection(A, B):
            c_a = collections.Counter(A)
            c_b = collections.Counter(B)
            print("Key value of node and number of times occur after sampling", c_a)
            a = c_a.keys()
            b = c_b.keys()
            c = a - b
            pp.append(c)
            duplicates = []
            for c in c_a:
                duplicates += [c] * min(c_a[c], c_b[c])
                different = c_a.keys() - duplicates

            if len(different) == 0:
                print("q is empty", 0)
                print("p is", len(flat_list))
            else:
                q_count = []
                for i in set(diff):
                    q_count.append(c_a.get(i))
                for g in flat_list:
                    if g in c_a.keys() & c_b.keys():
                        (c_a.keys() & c_b.keys()).remove(g)

            return duplicates
            print()

        intersection(flat_list, inter)

    print()
    for node in output:
        print("Sample set for target node {} is {}".format(node, output[node]))
    print()

    Proxy_node_flatten = [element for items in pp for element in items]
    print("Proxy node", set(Proxy_node_flatten))
    print("Length of proxy nodes", len(set(Proxy_node_flatten)))
