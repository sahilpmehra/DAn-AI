<script lang="ts">
    import Menu from "$lib/components/ui/icons/Menu.svelte";
    import MessageSquare from "$lib/components/ui/icons/MessageSquare.svelte";
    import Settings from "$lib/components/ui/icons/Settings.svelte";
    import Users from "$lib/components/ui/icons/Users.svelte";
    import X from "$lib/components/ui/icons/X.svelte";
    import { cn } from "$lib/utils";
    import { isFileUploaded } from "$lib/stores/stores";

    type TabItem = {
        icon: typeof MessageSquare | typeof Settings | typeof Users;
        label: string;
        id: string;
    }

    const tabs: TabItem[] = [
        { icon: MessageSquare, label: "Chat", id: "chat" },
        { icon: Settings, label: "Data Summary", id: "data-summary" },
        { icon: Users, label: "Key Variables", id: "visualizations" },
    ];

    let isOpen = $state(true);
    let { 
        class: className = '',
        currentRoute = '',
        onNavigate = (route: string) => {} 
    } = $props();
</script>

<div
    class={cn(
    "h-screen bg-sidebar border-r border-sidebar-border flex flex-col fixed left-0 w-64",
    className
    )}
>
    <div class="p-4 flex items-center border-b border-sidebar-border">
        <h2 class="font-semibold">
            AI Assistant
        </h2>
    </div>

    <nav class="flex-1 p-2">
        {#if $isFileUploaded}
            {#each tabs as tab}
                {@const Icon = tab.icon}
                <button
                    onclick={() => onNavigate(tab.id)}
                    class={cn(
                        "w-full flex items-center gap-3 p-3 rounded-md transition-colors",
                        "hover:bg-sidebar-accent",
                        currentRoute === tab.id && "bg-sidebar-accent text-sidebar-accent-foreground"
                    )}
                >
                    <Icon class="h-5 w-5" />
                    <span>{tab.label}</span>
                </button>
            {/each}
        {:else}
            <p class="text-sm text-muted-foreground p-3">
                Upload a file to begin analysis
            </p>
        {/if}
    </nav>
</div>
