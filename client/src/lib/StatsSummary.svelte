<script lang="ts">
  import {
    Card,
    CardContent,
    CardHeader,
    CardTitle,
    CardDescription,
  } from "$lib/components/ui/card";
  import DataTable from "$lib/components/DataTable.svelte";

  let { data, headers } = $props<{
    data: any[];
    headers: string[];
  }>();
</script>

<Card>
  <CardHeader>
    <CardTitle>Statistical Summary</CardTitle>
    <CardDescription>Summary of the numerical columns in the dataset</CardDescription>
  </CardHeader>
  <CardContent>
    <DataTable data={data} headers={headers} />
  </CardContent>
</Card>
