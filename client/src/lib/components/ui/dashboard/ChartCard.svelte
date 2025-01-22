<script lang="ts">
    import { Card } from "$lib/components/ui/card";
    import { Button } from "$lib/components/ui/button";
    import Download from "$lib/components/ui/icons/Download.svelte";
    
    let { title, children } = $props<{
        title: string;
        children: any;
    }>();
</script>

<Card class="p-4">
    <div class="flex justify-between items-center mb-4">
        <h3 class="font-medium">{title}</h3>
        <Button variant="ghost" size="icon">
            <Download class="h-4 w-4" />
        </Button>
    </div>
    <div class="aspect-video bg-gray-50 rounded-lg flex items-center justify-center">
        {@render children()}
    </div>
</Card>