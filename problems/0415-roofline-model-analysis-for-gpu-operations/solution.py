def roofline_analysis(peak_gflops: float, peak_bandwidth_gbs: float, operations: list) -> dict:
	"""
	Perform Roofline Model analysis for GPU operations.

	Args:
		peak_gflops: Peak compute throughput in GFLOPS
		peak_bandwidth_gbs: Peak memory bandwidth in GB/s
		operations: List of dicts with keys 'name', 'flops', 'bytes'

	Returns:
		Dict with 'ridge_point' and 'operations' list containing
		per-operation analysis results.
	"""
	ridge_point = peak_gflops/peak_bandwidth_gbs
	analysis = []
	for operation in operations:
		operation_analysis = {}
		operation_analysis['name'] = operation['name']
		operation_analysis['operational_intensity'] = operation['flops']/operation['bytes']
		operation_analysis['attainable_gflops'] = min(peak_gflops, peak_bandwidth_gbs*operation_analysis['operational_intensity'])
		if operation_analysis['operational_intensity']<ridge_point:
			operation_analysis['bottleneck'] = 'memory-bound'
		else:
			operation_analysis['bottleneck'] = 'compute-bound'
		operation_analysis['efficiency'] = 100*operation_analysis['attainable_gflops']/peak_gflops
		analysis.append(operation_analysis)
	return {'ridge_point': ridge_point, 'operations': analysis}