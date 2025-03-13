<script lang="ts">
	import { Button } from "$components";
</script>

<div class="relative isolate px-6 pt-14 lg:px-8">
	<div class="mx-auto max-w-2xl py-32 sm:py-48 lg:py-56">
		<div class="text-center">
			<h1
				class="text-5xl font-semibold tracking-tight text-balance text-red-950 sm:text-7xl"
			>
				Insurance Sell Prediction Project
			</h1>
			<p
				class="mt-8 text-lg font-medium text-pretty text-gray-500 sm:text-xl/8"
			>
				A python project to practice Machine Learning
				stacks.
			</p>
			<div
				class="mt-10 flex items-center justify-center gap-x-6"
			>
				<Button href="/dashboard"
					>Go to dashboard</Button
				>
				<Button
					href="#"
					isSecondary={true}
					isExternal={true}
					>Documentation <span aria-hidden="true"
						>→</span
					></Button
				>
			</div>
		</div>
	</div>
</div>
