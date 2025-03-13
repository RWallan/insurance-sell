<script lang="ts">
        import type { Snippet } from "svelte";

        interface BasicProps {
                children: Snippet;
                isSecondary?: boolean;
        }

        interface ButtonProps extends BasicProps {
                onclick?: (e: MouseEvent) => void;
                href?: never;
                type?: "button" | "submit";
                isExternal?: never;
                isMenu?: never;
        }

        interface LinkProps extends BasicProps {
                onclick?: never;
                href: string;
                type?: never;
                isMenu?: boolean;
                isExternal?: boolean;
        }

        type ComponentProps = ButtonProps | LinkProps;

        let {
                children,
                onclick,
                href,
                isExternal,
                isMenu,
                isSecondary,
                ...props
        }: ComponentProps = $props();
</script>

{#if href}
        <a
                {href}
                class={isSecondary
                        ? "text-sm/6 font-semibold text-gray-900"
                        : isMenu
                          ? "rounded-md px-3 py-2 text-sm font-medium text-gray-300 hover:bg-gray-700 hover:text-white"
                          : "rounded-md bg-gray-800 px-3.5 py-2.5 text-sm font-semibold text-white shadow-xs hover:bg-gray-600 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"}
                target={isExternal ? "_blank" : "_self"}>{@render children()}</a
        >
{:else}
        <button
                class={isSecondary
                        ? "text-sm/6 font-semibold text-gray-900"
                        : "rounded-md bg-gray-800 px-3.5 py-2.5 text-sm font-semibold text-white shadow-xs hover:bg-gray-600 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"}
                {...props}
                >{@render children()}
        </button>
{/if}
