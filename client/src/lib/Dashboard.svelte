<script lang="ts">
    import { ChartCard, SidePanel, TopPanel } from "$lib/components/ui/dashboard";
</script>

<div class="min-h-screen bg-gray-50">
    <div class="flex">
      <main class="flex-1 p-6">
        <div class="max-w-7xl mx-auto">
          <div class="flex justify-between items-center mb-6">
            <h1 class="text-2xl font-bold">Data Dashboard</h1>
            <button class="px-4 py-2 bg-dashboard-primary text-white rounded-lg hover:bg-blue-700 transition-colors">
              Export All
            </button>
          </div>

          <TopPanel />

          <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <ChartCard title="Distribution">
              <p class="text-gray-500">Bar Chart Placeholder</p>
            </ChartCard>
            <ChartCard title="Trend Analysis">
              <p class="text-gray-500">Line Chart Placeholder</p>
            </ChartCard>
            <ChartCard title="Composition">
              <p class="text-gray-500">Pie Chart Placeholder</p>
            </ChartCard>
            <ChartCard title="Correlation">
              <p class="text-gray-500">Scatter Plot Placeholder</p>
            </ChartCard>
          </div>
        </div>
      </main>
      <SidePanel />
    </div>
</div>