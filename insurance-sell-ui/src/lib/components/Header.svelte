<script lang="ts">
        import { Button } from "$components";
</script>

<nav class="bg-gray-800">
        <div class="flex justify-center mx-auto max-w-7xl px-2 sm:px-6 lg:px-8">
                <div class="relative flex h-16 items-center justify-between">
                        <div class="flex space-x-4">
                                <Button href="/" isMenu={true}>Home</Button>
                                <Button href="#" isMenu={true} isExternal={true}
                                        >Documentation <span
                                                aria-hidden="true"
                                                class="hidden sm:inline"
                                                >↗</span
                                        ></Button
                                >
                                <Button
                                        href="http://localhost:4200"
                                        isMenu={true}
                                        isExternal={true}
                                        >Pipelines <span
                                                aria-hidden="true"
                                                class="hidden sm:inline"
                                                >↗</span
                                        ></Button
                                >
                                <Button
                                        href="http://localhost:5000"
                                        isMenu={true}
                                        isExternal={true}
                                        >Model <span
                                                aria-hidden="true"
                                                class="hidden sm:inline"
                                                >↗</span
                                        ></Button
                                >
                        </div>
                </div>
        </div>
</nav>
