<script lang="ts">
    import {Table, TableBody, TableCell, TableHead, TableHeader, TableRow} from '$lib/components/ui/table';
    import {ScrollArea} from '$lib/components/ui/scroll-area';

    let { data, headers } = $props<{
        data: any[];
        headers: string[];
    }>();
</script>

<ScrollArea class="border rounded-lg max-h-[400px] overflow-y-auto">
    <Table>
        <TableHeader class="sticky top-0">
            <TableRow>
                {#each headers as header}
                    <TableHead class="text-center bg-gray-200">{header}</TableHead>
                {/each}
            </TableRow>
        </TableHeader>
        <TableBody>
            {#each data as row}
                <TableRow>
                    {#each headers as header, i}
                        <TableCell class="text-center {i % 2 === 0 ? 'bg-gray-100' : 'bg-white'}">{row[header]}</TableCell>
                    {/each}
                </TableRow>
            {/each}
        </TableBody>
    </Table>
</ScrollArea>