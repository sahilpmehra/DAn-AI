<script lang="ts">
  import { Card, CardContent, CardHeader, CardTitle } from "$lib/components/ui/card";
  import { Alert, AlertDescription } from "$lib/components/ui/alert";
  import TriangleAlert from "$lib/components/ui/icons/TriangleAlert.svelte";
  import { Button } from "$lib/components/ui/button";
  import { RadioGroup, RadioGroupItem } from "$lib/components/ui/radio-group";
  import { Label } from "$lib/components/ui/label";
  import { Checkbox } from "$lib/components/ui/checkbox";
  import { toast } from "$lib/hooks/use-toast";
  import { analysisConfigStore } from "$lib/stores/analysisStore";

  let { summary, keyVariables, problematicVariables } = $props<{
    summary: string;
    keyVariables: string[];
    problematicVariables: string[];
  }>();

  let decision = $state<"accept" | "reject" | "customize" | undefined>($analysisConfigStore.decision);
  let selectedKeyVars = $state<string[]>([...keyVariables]);
  let selectedProbVars = $state<string[]>([...problematicVariables]);

  function handleSubmit() {
    if (!decision) {
      toast({
        title: "Please make a selection",
        description: "Choose whether to accept, reject, or customize the recommendations.",
        variant: "destructive",
      });
      return;
    }

    const payload = {
      decision,
      keyVariables: decision === "customize" ? selectedKeyVars : keyVariables,
      problematicVariables: decision === "customize" ? selectedProbVars : problematicVariables,
    };
    
    console.log("Submitting to backend:", payload);
    toast({
      title: "Success!",
      description: "Your preferences have been saved.",
    });
    
    // Update the store
    analysisConfigStore.set({
      isConfigured: true,
      decision,
      selectedKeyVars,
      selectedProbVars
    });
  }

  function handleEdit() {
    analysisConfigStore.set({
      ...$analysisConfigStore,
      isConfigured: false
    });
  }
</script>

<Card>
  <CardHeader>
    <CardTitle class="flex justify-between items-center">
      Analysis Summary
      {#if $analysisConfigStore.isConfigured}
        <Button variant="outline" size="sm" onclick={handleEdit}>
          Edit Variables
        </Button>
      {/if}
    </CardTitle>
  </CardHeader>
  <CardContent class="space-y-6">
    <p class="text-gray-700">{summary}</p>
    
    <div>
      <h4 class="font-semibold mb-2">Selected Key Variables:</h4>
      <ul class="list-disc list-inside space-y-1">
        {#each (decision === "customize" ? selectedKeyVars : keyVariables) as variable (variable)}
          <li class="text-blue-600">{variable}</li>
        {/each}
      </ul>
    </div>

    {#if (decision === "customize" ? selectedProbVars : problematicVariables).length > 0}
      <Alert variant="destructive">
        <TriangleAlert class="h-4 w-4" />
        <AlertDescription>
          Excluded variables:
          <ul class="list-disc list-inside mt-2">
            {#each (decision === "customize" ? selectedProbVars : problematicVariables) as variable (variable)}
              <li>{variable}</li>
            {/each}
          </ul>
        </AlertDescription>
      </Alert>
    {/if}

    {#if !$analysisConfigStore.isConfigured}
      <div class="space-y-4 border-t pt-4">
        <h4 class="font-semibold">Would you like to accept these recommendations?</h4>
        <RadioGroup value={decision} onValueChange={(value) => decision = value as "accept" | "reject" | "customize"} class="space-y-2">
          <div class="flex items-center space-x-2">
            <RadioGroupItem value="accept" id="accept" />
            <Label for="accept">Yes, accept all recommendations</Label>
          </div>
          <div class="flex items-center space-x-2">
            <RadioGroupItem value="reject" id="reject" />
            <Label for="reject">No, I'll analyze without these recommendations</Label>
          </div>
          <div class="flex items-center space-x-2">
            <RadioGroupItem value="customize" id="customize" />
            <Label for="customize">I want to customize the selections</Label>
          </div>
        </RadioGroup>

        {#if decision === "customize"}
          <div class="space-y-4 mt-4 p-4 bg-gray-50 rounded-md">
            <div>
              <h5 class="font-medium mb-2">Select Key Variables:</h5>
              <div class="space-y-2">
                {#each keyVariables as variable (variable)}
                  <div class="flex items-center space-x-2">
                    <Checkbox
                      id={`key-${variable}`}
                      checked={selectedKeyVars.includes(variable)}
                      onCheckedChange={(checked) => {
                        if (checked) {
                          selectedKeyVars = [...selectedKeyVars, variable];
                        } else {
                          selectedKeyVars = selectedKeyVars.filter((v) => v !== variable);
                        }
                      }}
                    />
                    <Label for={`key-${variable}`}>{variable}</Label>
                  </div>
                {/each}
              </div>
            </div>

            <div>
              <h5 class="font-medium mb-2">Select Variables to Exclude:</h5>
              <div class="space-y-2">
                {#each problematicVariables as variable (variable)}
                  <div class="flex items-center space-x-2">
                    <Checkbox
                      id={`prob-${variable}`}
                      checked={selectedProbVars.includes(variable)}
                      onCheckedChange={(checked) => {
                        if (checked) {
                          selectedProbVars = [...selectedProbVars, variable];
                        } else {
                          selectedProbVars = selectedProbVars.filter((v) => v !== variable);
                        }
                      }}
                    />
                    <Label for={`prob-${variable}`}>{variable}</Label>
                  </div>
                {/each}
              </div>
            </div>
          </div>
        {/if}

        <Button 
          onclick={handleSubmit}
          class="mt-4"
        >
          Confirm Selection
        </Button>
      </div>
    {/if}
  </CardContent>
</Card>
