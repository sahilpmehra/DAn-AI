<script lang="ts">
    import { Card } from "$lib/components/ui/card";
    import { Button } from "$lib/components/ui/button";
    import Download from "$lib/components/ui/icons/Download.svelte";
    
    let { title, children, class: className = '' } = $props<{
        title: string;
        children: any;
        class?: string;
    }>();
</script>

<Card class={className + " p-4"}>
    <div class="flex justify-between items-center mb-4">
        <h3 class="text-xl font-medium">{title}</h3>
        <Button variant="ghost" size="icon">
            <Download class="h-4 w-4" />
        </Button>
    </div>
    <div class="bg-gray-50 rounded-lg">
        {@render children()}
    </div>
</Card>