<script lang="ts">
    import { ChartCard, SidePanel, TopPanel } from "$lib/components/ui/dashboard";
    import { Tabs, TabsList, TabsTrigger, TabsContent } from "$lib/components/ui/tabs";

    type Tab = {
        value: string;
        label: string;
        chartType: string;
    };

    let tabs = $state<Tab[]>([
        { value: 'trends', label: 'Trends', chartType: 'Line Chart' },
        { value: 'comparison', label: 'Comparison', chartType: 'Bar Chart' },
        { value: 'composition', label: 'Composition', chartType: 'Pie Chart' },
        { value: 'correlation', label: 'Correlation', chartType: 'Scatter Plot' }
    ]);

    let activeTab = $state('trends');
</script>

<div class="min-h-screen bg-gray-50">
    <div class="flex">
      <main class="flex-1 p-6">
        <div class="max-w-7xl mx-auto">
          <div class="flex justify-between items-center mb-6">
            <h1 class="text-2xl font-bold">Data Dashboard</h1>
            <button class="px-4 py-2 bg-dashboard-primary rounded-lg hover:bg-blue-700 hover:text-white transition-colors">
              Export All
            </button>
          </div>

          <TopPanel />

          <div class="gap-6">
            <ChartCard title="Analytics Overview">
              {#if tabs.length > 1}
              <Tabs value={activeTab}>
                <TabsList class="flex flex-row">
                  {#each tabs as tab}
                      <TabsTrigger value={tab.value} class="flex-1">{tab.label}</TabsTrigger>
                  {/each}
                </TabsList>
                {#each tabs as tab}
                    <TabsContent value={tab.value}>
                        <p class="text-gray-500">{tab.chartType} Placeholder</p>
                    </TabsContent>
                {/each}
              </Tabs>
              {:else if tabs.length === 1}
                <p class="text-gray-500">{tabs[0].chartType} Placeholder</p>
              {:else}
                <p class="text-gray-500">No tabs available</p>
              {/if}
            </ChartCard>
          </div>
        </div>
      </main>
      <SidePanel />
    </div>
</div>