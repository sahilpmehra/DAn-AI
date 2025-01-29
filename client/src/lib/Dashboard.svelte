<script lang="ts">
    import { ChartCard, SidePanel, TopPanel } from "$lib/components/dashboard";
    import { Tabs, TabsList, TabsTrigger, TabsContent } from "$lib/components/ui/tabs";
    import { LineChart, BarChart, PieChart, ScatterPlot } from "$lib/components/charts";

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

    // Sample data for each chart
    const lineChartData = {
        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        values: [65, 59, 80, 81, 56, 55],
        label: 'Monthly Sales',
        xAxisLabel: 'Month',
        yAxisLabel: 'Sales'
    };

    const barChartData = {
        labels: ['Product A', 'Product B', 'Product C', 'Product D', 'Product E'],
        values: [120, 90, 60, 180, 45],
        label: 'Product Performance',
        xAxisLabel: 'Product',
        yAxisLabel: 'Sales'
    };

    const pieChartData = {
        labels: ['Desktop', 'Mobile', 'Tablet'],
        values: [450, 300, 150],
        title: 'Traffic Sources'
    };

    const scatterChartData = {
        points: [
            { x: 25, y: 30000 },
            { x: 30, y: 45000 },
            { x: 35, y: 55000 },
            { x: 40, y: 70000 },
            { x: 45, y: 85000 },
            { x: 50, y: 95000 },
            { x: 55, y: 100000 },
        ],
        label: 'Age vs. Income Distribution',
        xAxisLabel: 'Age',
        yAxisLabel: 'Income ($)'
    };
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
                        {#if tab.value === 'trends'}
                            <LineChart class="bg-gray-800 rounded-xl" data={lineChartData} />
                        {:else if tab.value === 'comparison'}
                            <BarChart class="bg-gray-800 rounded-xl" data={barChartData} />
                        {:else if tab.value === 'composition'}
                            <PieChart class="bg-gray-800 rounded-xl" data={pieChartData} />
                        {:else if tab.value === 'correlation'}
                            <ScatterPlot class="bg-gray-800 rounded-xl" data={scatterChartData} />
                        {/if}
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