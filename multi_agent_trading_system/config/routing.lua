-- MATS ToolHive VirtualMCP routing rules.
--
-- Input:  request:header(name), request:tool_namespace(), request:tool_name()
-- Output: a backend name matching one of the servers in virtual-mcp.yaml.

local function route(request)
  local namespace = request:tool_namespace()
  local name = request:tool_name()

  -- Memory tier routing
  if namespace == "obsidian" or namespace == "mats-vault" then
    return "obsidian-backend"
  end
  if namespace == "scs" then
    return "scs-backend"
  end
  if namespace == "policy" then
    return "policy-backend"
  end

  -- Trading venues
  if namespace == "coinbase" then
    return "coinbase-backend"
  end
  if namespace == "wasabi" then
    return "wasabi-backend"
  end
  if namespace == "predictbase" then
    return "predictbase-backend"
  end
  if namespace == "base" then
    return "base-backend"
  end

  -- Rewards + predictions
  if namespace == "allora" then
    return "allora-backend"
  end
  if namespace == "venice" then
    return "venice-backend"
  end

  -- Default: composite
  return "composite-backend"
end

return route
