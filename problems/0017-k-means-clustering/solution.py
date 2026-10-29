def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
	dim = len(points[0])
	final_centriods = initial_centroids.copy()
	for iteration in range(max_iterations):
		clusters = [[] for _ in range(k)]
		
		for point in points:
			min_dist, cluster_id = float("Inf"), None
			for i, centriod in enumerate(final_centriods):
				dist = sum([(xip-xic)**2 for xip,xic in zip(point, centriod)])
				if dist<=min_dist:
					min_dist = dist
					cluster_id = i
			clusters[cluster_id].append(point)

		new_centriods = []
		for k_ in range(k):
			if len(clusters[k_])==0:
				new_centriods.append([])
			else:
				centriod = []
				for i in range(dim):
					xic = sum([x[i] for x in clusters[k_]])/len(clusters[k_])
					centriod.append(xic)
				new_centriods.append(tuple(centriod))
		final_centriods = new_centriods

	return final_centriods