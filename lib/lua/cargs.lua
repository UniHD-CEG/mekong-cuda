local M = {}

--------------------------------------------------------------------------------
-- "forward declarations"
--------------------------------------------------------------------------------

local App = {}
local Option = {}

--------------------------------------------------------------------------------
-- APP IMPLEMENTATION
--
-- App object:
-- name: application name
-- help: application description
-- metavar: text shown for additional arguments
--
-- options: { Option }
--
-- add_option: (self, option) -> ()
-- parse: (args or nil) -> ( { <name> = <value> }, { <arguments> })
-- help: (self) -> ()
--------------------------------------------------------------------------------

App.__index = App
M.App = App

function App:new(name, description, metavar)
  local new = setmetatable({}, App)

  new.name = name
  new.description = description
  new.metavar = metavar or '<arguments>'

  new.options = {}

  new:add_option('flag', 'last', {'-', '--'}, 'help', nil, 'Show this help')

  return new
end

function App:add_option(parser, merger, prefixes, pattern, name, help, metavar)
  local option = Option:new(parser, merger, prefixes, pattern, name, help, metavar)
  self.options[#self.options + 1] = option
end

function App:parse(args)
  local options = {}
  local arguments = {}

  args = args or rawget(_G, "arg") or {}
  local err, idx = nil, 1
  while idx <= #args do
    if args[idx] == '--' then
      idx = idx + 1
      break
    end
    local is_option = false
    for _, option in ipairs(self.options) do
      local akk = options[option.name] or option.initial
      err, akk, idx = option:try(akk, idx, args)
      if err ~= nil then
        print("option " .. option.prefixes[1] .. option.pattern .. err)
        self:help()
        os.exit(1)
      end
      if akk ~= nil then
        is_option = true
        options[option.name] = akk
        break
      end
    end
    if not is_option then
      if args[idx]:sub(1, 1) == '-' then
        print('unknown option: ' .. args[idx])
        os.exit(1)
      end
      arguments[#arguments+1] = args[idx]
      idx = idx + 1
    end
  end
  while idx <= #args do
    arguments[#arguments+1] = args[idx]
    idx = idx + 1
  end
  if options['help'] then
    self:help()
    os.exit(0)
  end
  return options, arguments
end

function App:help()
  local printf = function(fmt, ...) print(string.format(fmt, ...)) end
  local rpad = function(str, n) return str .. string.rep(' ', n - #str+1) end
  printf("%s", self.description)
  printf("")
  printf("usage: %s [options] %s", self.name, self.metavar)
  printf("")
  printf("options:")
  local max_length = 0
  for _, option in ipairs(self.options) do
    local text = option.pattern
    if option.parser ~= 'flag' then
      text = text .. ' ' .. option.metavar
    end
    if #text > max_length then
      max_length = #text
    end
  end
  for _, option in ipairs(self.options) do
    if option.parser == 'flag' then
      local pattern = option.prefixes[1] .. option.pattern
      printf("  %s  %s", rpad(pattern, max_length), option.help)
    elseif option.parser == 'separate' or option.parser == 'joined_or_separate' then
      local pattern = option.prefixes[1] .. option.pattern .. ' ' .. option.metavar
      printf("  %s  %s", rpad(pattern, max_length), option.help)
    elseif option.parser == 'joined' then
      local pattern = option.prefixes[1] .. option.pattern .. option.metavar
      printf("  %s  %s", rpad(pattern, max_length), option.help)
    end
  end
end

--------------------------------------------------------------------------------
-- OPTION IMPLEMENTATION
--
-- Option object:
--   prefixes: prefixes for pattern
--   pattern: pattern to match agains
--   initial: initial value
--   name: name in global list
--   help: description of option
--   metavar: variable name shown in help
--
--   parser: 'flag' | 'joined' | 'separate' | 'joined_or_separate'
--   merger: nil | 'once' | 'last' | 'list'
--
-- try:
-- (self, akk, idx, args) ->
--   on match: nil, akk, idx
--   on fail: nil, old-idx
--   on error: error
--------------------------------------------------------------------------------

Option.__index = Option
M.Option = Option

function Option:new_kw(options)
  local new = setmetatable({}, Option)

  new.parser = options.parser
  new.merger = options.merger or 'once'
  new.prefixes = options.prefixes or {'-', '--'}
  new.pattern = options.pattern
  new.name = options.name or new.pattern
  new.help = options.help or 'no help available'
  new.metavar = options.metavar or '<value>'

  assert(M.parsers[new.parser], 'invalid parser: "' .. (new.parser or 'nil') .. '"')
  assert(M.mergers[new.merger], 'invalid merger: "' .. new.merger .. '"')
  if type(new.prefixes) ~= 'table' then error('prefixes must be a list') end
  if type(new.pattern) ~= 'string' then error('pattern must be a string') end

  return new
end

function Option:new(parser, merger, prefixes, pattern, name, help, metavar)
  local new = setmetatable({}, Option)

  new.parser = parser
  new.merger = merger or 'once'
  new.prefixes = prefixes or {'-', '--'}
  new.pattern = pattern
  new.name = name or new.pattern
  new.help = help or 'no help available'
  new.metavar = metavar or '<value>'

  assert(M.parsers[new.parser], 'invalid parser: "' .. (new.parser or 'nil') .. '"')
  assert(M.mergers[new.merger], 'invalid merger: "' .. new.merger .. '"')
  if type(new.prefixes) ~= 'table' then error('prefixes must be a list') end
  if type(new.pattern) ~= 'string' then error('pattern must be a string') end

  return new
end

function Option:try(akk, idx, args)
  local parse = M.parsers[self.parser]
  local merge = M.mergers[self.merger]

  for _, prefix in ipairs(self.prefixes) do
    local err, val, newidx = parse(prefix .. self.pattern, idx, args)
    if err ~= nil then
      return err
    end
    if val ~= nil then
      akk  = akk or self.initial
      err, akk = merge(akk, val)
      if err ~= nil then
        return err
      else
        return nil, akk, newidx
      end
    end
  end
  return nil, nil, idx
end

--------------------------------------------------------------------------------
-- Combiners
-- interface: (akk, value) -> err, new akk
--------------------------------------------------------------------------------

function M.merge_once(akk, value)
  if akk ~= nil then
    return "can only be used once", nil
  else
    return nil, value
  end
end

function M.merge_last(akk, value)
  return nil, value
end

function M.merge_list(akk, value)
  local new = {}
  akk = akk or {}
  for idx, val in ipairs(akk) do
    new[idx] = val
  end
  new[#new+1] = value
  return nil, new
end

M.mergers = {
  ['once'] = M.merge_once,
  ['last'] = M.merge_last,
  ['list'] = M.merge_list,
}

--------------------------------------------------------------------------------
-- PARSER IMPLEMENTATIONS
--
-- interface:
-- (pattern, position, argument list) ->
--   if matched: nil, value, new position
--   if failed: nil, nil, old position
--   if error: error
--
-- err OR nil, flag/option value OR nil, new idx
--------------------------------------------------------------------------------

function M.parse_flag(pattern, idx, args)
  if args[idx] == pattern then
    return nil, true, idx + 1
  else
    return nil, nil, idx
  end
end

function M.parse_joined(pattern, idx, args)
  if args[idx]:sub(1, #pattern) == pattern then
    return nil, args[idx]:sub(#pattern+1), idx + 1
  else
    return nil, nil, idx
  end
end

function M.parse_separate(pattern, idx, args)
  if args[idx] == pattern then
    if #args < idx + 1 then
      return "requires value"
    else
      return nil, args[idx+1], idx + 2
    end
  else
    return nil, nil, idx
  end
end

-- does not reuse parse_joined/parse_separate because we need a new special case
-- anyway: parse_joined succeeds at "-I" with value "", which is invalid
function M.parse_joined_or_separate(pattern, idx, args)
  if args[idx]:sub(1, #pattern) == pattern then
    if #args[idx] == #pattern then -- separate
      if #args < idx + 1 then
        return "requires value"
      else
        return nil, args[idx+1], idx + 2
      end
    else -- joined
      return nil, args[idx]:sub(#pattern+1), idx + 1
    end
  else
    return nil, nil, idx
  end
end

M.parsers = {
  ['flag'] = M.parse_flag,
  ['joined'] = M.parse_joined,
  ['separate'] = M.parse_separate,
  ['joinsep'] = M.parse_joined_or_separate,
  ['joined_or_separate'] = M.parse_joined_or_separate,
}

return M
