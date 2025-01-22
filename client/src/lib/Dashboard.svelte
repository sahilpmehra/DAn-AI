<script lang="ts">
    import { ChartCard, SidePanel, TopPanel } from "$lib/components/ui/dashboard";
    import { Root, List, Trigger, Content } from "$lib/components/ui/tabs";
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

          <div class="gap-6">
            <ChartCard title="Analytics Overview">
              <Root value="trends">
                <List>
                  <Trigger value="trends">Trends</Trigger>
                  <Trigger value="comparison">Comparison</Trigger>
                  <Trigger value="composition">Composition</Trigger>
                  <Trigger value="correlation">Correlation</Trigger>
                </List>
                <Content value="trends">
                  <p class="text-gray-500">Line Chart Placeholder</p>
                </Content>
                <Content value="comparison">
                  <p class="text-gray-500">Bar Chart Placeholder</p>
                </Content>
                <Content value="composition">
                  <p class="text-gray-500">Pie Chart Placeholder</p>
                </Content>
                <Content value="correlation">
                  <p class="text-gray-500">Scatter Plot Placeholder</p>
                </Content>
              </Root>
            </ChartCard>
          </div>
        </div>
      </main>
      <SidePanel />
    </div>
</div>