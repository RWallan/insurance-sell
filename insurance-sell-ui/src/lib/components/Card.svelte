<script lang="ts">
        interface CardProps {
                title: string;
                value: number;
        }

        let { title, value }: CardProps = $props();
</script>

<div class="max-w-xs rounded overflow-hidden shadow-lg px-6 py-4">
        <p class="font-semibold text-red-950 text-base">{title}</p>
        <h1 class="font-bold text-2xl text-gray-800">{value}</h1>
</div>
