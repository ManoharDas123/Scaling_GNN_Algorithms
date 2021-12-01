def optimized_sampling_function(G, alpha, k, partition_sets, xadj, adjncy, sample_count=15):
    D_list = []

    for p in range(len(partition_sets)):
        D_list.append(set())

    print("Dlist is",D_list)
    for node in range(len(G.nodes)):
        neighbors_set = adjncy[xadj[node]:(xadj[node + 1])]
        node_in_partition_set = None
        same_partition_set_index = None

        for partition_index in range(len(partition_sets)):
            if node in partition_sets[partition_index]:
                node_in_partition_set = partition_sets[partition_index]
                same_partition_set_index = partition_index
                break

        same_partition_dynamic_set = D_list[same_partition_set_index]
        same_partition_dynamic_size = len(same_partition_dynamic_set)

        inter = node_in_partition_set.intersection(neighbors_set)
        x = len(inter)

        diff = set(neighbors_set) - inter
        y = len(diff)
        w1 = []
        w2 = []

        for i in range(x):
            w1.append(alpha / 10 / (x + same_partition_dynamic_size))

        for i in range(y):
            w2.append((alpha / 10 / (x + same_partition_dynamic_size)) / (len(neighbors_set)))

        sample_picks = []
        for i in range(sample_count):
            sample_pick = random.choices(population=list(inter) + list(diff), weights=list(w1) + list(w2), k=k)
            sample_picks.append(sample_pick)

            for sampled_node in sample_pick:
                if sampled_node not in node_in_partition_set:
                    D_list[same_partition_set_index].add(sampled_node)

            own_partition_count = 0
            for i in sample_pick:
                if i in node_in_partition_set:
                    own_partition_count += 1

            if node in output:
                output[node].append(sample_pick)
            else:
                output[node] = [sample_pick]

        flat_list = [item for sublist in sample_picks for item in sublist]

        def intersection(A, B):
            c_a = collections.Counter(A)
            c_b = collections.Counter(B)

            a = c_a.keys()
            b = c_b.keys()
            c = a - b
            pp.append(c)

            duplicates = []
            for c in c_a:
                duplicates += [c] * min(c_a[c], c_b[c])
                different = c_a.keys() - duplicates

        intersection(flat_list, inter)

    flatten1 = [element for items in proxy_node_set for element in items]

    Proxy_node_flatten = [element for items in pp for element in items]
    print("proxy length of optimized", len(D_list[0]) + len(D_list[1]))
