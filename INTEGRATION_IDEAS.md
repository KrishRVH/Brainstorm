# Potential Immolate Integration Improvements

## Current Limitations
- Balatro's Lua sandbox prevents direct GPU access
- OpenCL requires system-level permissions
- Game loop can't handle async GPU operations
- FFI limited to simple DLL calls

## Possible Enhancements

### 1. Semi-Integrated Approach
Create a helper script that:
- Launches Immolate.exe in background when Balatro starts
- Communicates via temp files
- Mod polls for results

```lua
-- In Brainstorm.lua
function Brainstorm.start_gpu_helper()
    -- Write search request to temp file
    nativefs.write("temp/search_request.txt", search_params)
    -- Helper monitors this file
    -- Results written to temp/search_results.txt
end
```

### 2. Hybrid DLL
Enhance current Immolate.dll to:
- Handle simple Erratic checks (< 100 seeds)
- Fall back to game restarts for complex searches
- Cache results for common patterns

### 3. Pre-Search Workflow Tool
Create a standalone GUI that:
- Runs before playing Balatro
- Searches and saves good seeds
- Exports to a file the mod can read

```lua
-- Mod reads pre-searched seeds
Brainstorm.good_seeds = nativefs.load("good_seeds.json")
```

### 4. IPC Communication
Use named pipes or sockets:
- Immolate.exe runs as service
- Mod communicates via local socket
- Async results without freezing

## Recommended Approach

For now, the best enhancement would be:

1. **Create launcher script** (`BrainstormLauncher.bat`):
```batch
@echo off
start /min ImmolateService.exe --listen
start Balatro.exe
```

2. **Add service mode to Immolate**:
```c
// ImmolateService mode
while (true) {
    request = read_file("requests.txt");
    if (request) {
        results = search_seeds(request);
        write_file("results.txt", results);
    }
    sleep(100ms);
}
```

3. **Mod polls for results**:
```lua
function Brainstorm.check_gpu_results()
    local results = nativefs.read("results.txt")
    if results then
        -- Process GPU search results
        Brainstorm.process_seeds(results)
    end
end
```

This would give near-seamless integration without fighting Balatro's sandbox.