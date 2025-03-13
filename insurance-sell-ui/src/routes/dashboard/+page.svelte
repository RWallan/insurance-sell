<script lang="ts">
	import { Card } from "$components";

	let { data } = $props();
	let trainMetrics = data.trainMetrics;
	let testMetrics = data.testMetrics;
</script>

<div class="flex justify-evenly mt-6">
	<div>
		<h1 class="text-3xl text-center text-red-950">Train Metrics</h1>
		<div
			class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-6 p-6"
		>
			{#each trainMetrics.metrics as metric}
				<Card
					title={metric.metric}
					value={metric.value}
				/>
			{/each}
		</div>
	</div>
	<div>
		<h1 class="text-3xl text-center text-red-950">Test Metrics</h1>
		<div
			class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-6 p-6"
		>
			{#each testMetrics.metrics as metric}
				<Card
					title={metric.metric}
					value={metric.value}
				/>
			{/each}
		</div>
	</div>
</div>
