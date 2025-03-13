<script lang="ts">
	import { Footer, Header } from "$components";
	let { children } = $props();
</script>

<Header />
{@render children()}
<Footer />
