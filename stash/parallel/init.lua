----------------------------------------------------------------------
--
-- Copyright (c) 2011 Clement Farabet
--
-- Permission is hereby granted, free of charge, to any person obtaining
-- a copy of this software and associated documentation files (the
-- "Software"), to deal in the Software without restriction, including
-- without limitation the rights to use, copy, modify, merge, publish,
-- distribute, sublicense, and/or sell copies of the Software, and to
-- permit persons to whom the Software is furnished to do so, subject to
-- the following conditions:
--
-- The above copyright notice and this permission notice shall be
-- included in all copies or substantial portions of the Software.
--
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
-- EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
-- MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
-- NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
-- LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
-- OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
-- WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
--
----------------------------------------------------------------------
-- description:
--     parallel - a package that provides a simple mechanism to
--                dispatch Lua scripts as independant processes
--                and communicate via a super raw shared memory
--                buffer
--
-- history:
--     September  2, 2011, 5:42PM - using 0MQ instead of IPC - Scoffier / Farabet
--     August 27, 2011, 6:31PM - beta release - Clement Farabet
--     August 26, 2011, 6:03PM - creation - Clement Farabet
----------------------------------------------------------------------

require 'os'
require 'io'
require 'sys'
require 'torch'
local zmq = require 'libluazmq'

local glob = _G
local assignedid
local assignedid = parallel and parallel.id or nil
local assignedip = parallel and parallel.ip or nil
local assignedparent = parallel and parallel.parent or nil
parallel = {}
parallel.zmq = zmq
local sys = sys
local type = type
local tostring = tostring
local torch = torch
local error = error
local table = table
local require = require
local os = os
local io = io
local pairs = pairs
local ipairs = ipairs
local package = package

--------------------------------------------------------------------------------
-- 0MQ context and options
--------------------------------------------------------------------------------
local zmqctx = zmq.init(1)
local currentport = 6000
parallel.currentport = currentport
parallel.zmqctx = zmqctx

local autoip, run, fork, nfork, sfork, exec, join, yield, sync, send, receive, close, print, addremote, calibrate, _fill, reset
--------------------------------------------------------------------------------
-- configure local IP
--------------------------------------------------------------------------------
autoip = function(interface)
            local interfaces
            if type(interface) == 'table' then
               interfaces = interface
            elseif interface then
               interfaces = {interface}
            end
            if sys.OS == 'linux' then
               interfaces = interfaces or {'eth0','eth1','wlan0','wlan1'}
               local ipfound
               for _,interface in ipairs(interfaces) do
                  ipfound = sys.fexecute("/sbin/ifconfig " .. interface
                                        .. " | grep 'inet addr:'| grep -v '127.0.0.1'"
                                        .. " | cut -d: -f2 | awk '{ print $1}'")
                  if ipfound:find('%d') then
                     parallel.ip = ipfound:gsub('%s','')
                     break
                  end
               end
            elseif sys.OS == 'macos' then
               interfaces = interfaces or {'en0','en1'}
               local ipfound
               for _,interface in ipairs(interfaces) do
                  ipfound = sys.fexecute("/sbin/ifconfig " .. interface
                                        .. " | grep -E 'inet.[0-9]' | grep -v '127.0.0.1'"
                                        .. " | awk '{ print $2}'")
                  if ipfound:find('%d') then
                     parallel.ip = ipfound:gsub('%s','')
                     break
                  end
               end
            else
               print('WARNING: unsupported OS')
               return
            end
         end
parallel.autoip = autoip
--------------------------------------------------------------------------------
-- run is a shortcut for fork/exec code on the local machine
--------------------------------------------------------------------------------
run = function(code,...)
         -- (1) fork process
         local child = fork(nil, nil, nil, ...)

         -- (2) exec code
         child:exec(code)
      end
parallel.run = run
--------------------------------------------------------------------------------
-- fork new idle process
--------------------------------------------------------------------------------
fork = function(rip, protocol, rlua, ...)
          -- (0) remote or local connection
          local lip
    local bin_name = jit and 'luajit' or 'lua'
          rlua = rlua or bin_name
          if rip then
             protocol = protocol or 'ssh -Y'
             if parallel.ip == '127.0.0.1' then
                print('<parallel.fork> WARNING: local ip is set to localhost, forked'
                      .. ' remote processes will not be able to reach it,'
                      .. ' please set your local ip: parallel.ip = "XX.XX.XX.XX"')
             end
             lip = parallel.ip
          else
             lip = '127.0.0.1'
          end

          -- (1) create sockets to communicate with child
          local sockwr = zmqctx:socket(parallel.zmq.PUSH)
          local sockrd = zmqctx:socket(parallel.zmq.PULL)
          local portwr = currentport
          while not sockwr:bind("tcp://*:" .. portwr) do
             currentport = currentport + 1
             portwr = currentport
          end
          local portrd = currentport
          while not sockrd:bind("tcp://*:" .. portrd) do
             currentport = currentport + 1
             portrd = currentport
          end

          -- (2) generate code for child
          --     this involve setting its id, parent id, and making sure it connects
          --     to its parent
          local str = ""
          str = str .. 'require([[package]]) '
          -- str = str .. "package.path = [[" .. package.path .. "]] "
          -- str = str .. "package.cpath = [[" .. package.cpath .. "]] "
          -- str = str .. "require [[env]]"
          str = str .. " loadstring = loadstring or load "
          str = str .. "parallel = {} "
          str = str .. "parallel.id = " .. parallel.processid .. " "
          str = str .. "parallel.parent = {id = " .. parallel.id .. "} "
          str = str .. "parallel = require([[parallel]]) "

          str = str .. "parallel.parent.socketrd = parallel.zmqctx:socket(parallel.zmq.PULL) "
          str = str .. "parallel.parent.socketrd:connect([[tcp://"..lip..":"..portwr.."]]) "
          str = str .. "parallel.parent.socketwr = parallel.zmqctx:socket(parallel.zmq.PUSH) "
          str = str .. "parallel.parent.socketwr:connect([[tcp://"..lip..":"..portrd.."]]) "
          local args = {...}
          str = str .. "parallel.args = {}"
          for i = 1,glob.select('#',...) do
             str = str .. 'table.insert(parallel.args, ' .. tostring(args[i]) .. ') '
          end
          str = str .. "_exec_ = parallel.parent:receive() "

          
          str = str .. [[for _,func in ipairs(_exec_) do 
                              local f = loadstring(func)
                              debug.setupvalue(f, 1, _ENV)
                              f()
                         end]]
          -- print(str)

          -- (3) fork a lua process, running the code dumped above
          local pid
          if protocol then
             pid = sys.fexecute(protocol .. ' ' .. rip ..
                               ' "' .. rlua .. " -e '" .. str .. "' " .. '" &  echo $!', '*line')
             -- print(protocol .. ' ' .. rip ..
             --                   ' "' .. rlua .. " -e '" .. str .. "' " .. '" &  echo $!')
          else
             pid = sys.fexecute(rlua .. ' -e "' .. str .. '" & echo $!', '*line')
          end
          pid = pid:gsub('%s','')

          -- (4) register child process for future reference
          local child = {id=parallel.processid, unixid=pid, ip=rip, speed=1,
                         socketwr=sockwr, socketrd=sockrd}
          _fill(child)
          parallel.children[parallel.processid] = child
          parallel.nchildren = parallel.nchildren + 1

          -- (5) incr counter for next process
          parallel.processid = parallel.processid + 1
          return child
       end
parallel.fork = fork
--------------------------------------------------------------------------------
-- nfork = fork N processes, according to the given configuration
-- the configuration is a table with N entries, each entry being:
-- entry = {NB_PROCESSES, ip='IP_ADDR', protocol='PROTOCOL', lua='REMOTE_LUA_CMD_LINE'}
--------------------------------------------------------------------------------
nfork = function(...)
           local args = {...}
           local config
           if type(args[1]) == 'table' then 
              config = args
              if type(config[1][1]) == 'table' then config = config[1] end
           else 
              config = {args} 
           end
           local forked = {}
           for i,entry in ipairs(config) do
              for k = 1,entry[1] do
                 local child = fork(entry.ip, entry.protocol, entry.lua)
                 table.insert(forked, child)
              end
           end
           _fill(forked)
           return forked
        end
parallel.nfork = nfork
--------------------------------------------------------------------------------
-- sfork = smart fork N processes, according to the current remotes table
-- parallel.addremote() should be called first to configure which machines are
-- available, and how many cores each one has
--------------------------------------------------------------------------------
sfork = function(nb)
           if not remotes then
              -- local fork
              return nfork(nb)
           else
              local forked = {}
              -- remote fork: distribute processes on all remotes
              while nb ~= 0 do
                 for i,remote in ipairs(remotes) do
                    if remote.cores > 0 or remotes.cores <= 0 then
                       local child = fork(remote.ip, remote.protocol, remote.lua)
                       child.remote = remote
                       child.speed = remote.speed or 1
                       remote.cores = remote.cores - 1
                       remotes.cores = remotes.cores - 1
                       table.insert(forked, child)
                       if remotes.cores < 0 then
                          print('WARNING: forking more processes than cores available')
                       end
                       nb = nb - 1
                       if nb == 0 then break end
                    end
                 end
              end
              _fill(forked)
              return forked
           end
        end
parallel.sfork = sfork
--------------------------------------------------------------------------------
-- exec code in given process
--------------------------------------------------------------------------------
exec = function(process, code)
          local processes = process
          if process.id then processes = {process} end
          -- make sure no process is already running code
          for _,process in pairs(processes) do
             if type(process) == 'table' then
                if process.running then
                   error('<parallel.exec> process already running code, cannot exec again')
                end
                process.running = true
             end
          end
          -- the code might be a function
          local exec = {}
          if glob.type(code) == 'function' then
             table.insert(exec, glob.string.dump(code))
          else
             table.insert(exec, code)
          end
          -- close() after code is executed
          table.insert(exec, glob.string.dump(function() parallel.close() end))
          -- load all processes with code
          send(processes, exec)
       end
parallel.exec = exec
--------------------------------------------------------------------------------
-- join = synchronize processes that have yielded, blocking call
--------------------------------------------------------------------------------
join = function(process, msg)
          msg = msg or ''
          if not process.id then 
             -- a list of processes to join
             for _,proc in pairs(process) do
                if type(proc) == 'table' then
                   proc.socketwr:send(msg)
                   proc.socketrd:recv()
                end
             end
          else 
             -- a single process to join
             process.socketwr:send(msg)
             process.socketrd:recv()
          end
       end
parallel.join = join
--------------------------------------------------------------------------------
-- yield = interupt execution flow to allow parent to join
--------------------------------------------------------------------------------
yield = function()
           local msg = parallel.parent.socketrd:recv()
           parallel.parent.socketwr:send('!')
           return msg
        end
parallel.yield = yield
--------------------------------------------------------------------------------
-- sync = wait on a process, or processes, to terminate
--------------------------------------------------------------------------------
sync = function(process)
          if not process.id then 
             -- a list of processes to sync
             for _,proc in pairs(process) do
                if type(proc) == 'table' then
                   sync(proc)
                end
             end
          else 
             -- a single process to sync
             while true do
                local alive = sys.fexecute("ps -ef | awk '{if ($2 == " .. 
                                          process.unixid .. ") {print $2}}'"):gsub('%s','')
                if alive == '' then
                   if process.remote and process.remote.cores then
                      process.remote.cores = process.remote.cores + 1
                      remotes.cores = remotes.cores + 1
                   end
                   parallel.children[process.id] = nil
                   parallel.nchildren = parallel.nchildren - 1
                   break
                end
             end
          end
       end
parallel.sync = sync
--------------------------------------------------------------------------------
-- transmit data
--------------------------------------------------------------------------------
send = function(process, object)
          if not process.id then 
             -- multiple processes
             local processes = process
             -- a list of processes to send data to
             if not (torch.typename(object) and torch.typename(object):find('torch.*Storage')) then
                -- serialize data once for all transfers
                local f = torch.MemoryFile()
                f:binary()
                f:writeObject(object)
                object = f:storage()
                f:close()
             end

             -- broadcast storage to all processes
             for _,process in pairs(processes) do
    
                if type(process) == 'table' then
                  -- print(object)
                   object.zmq.send(object, process.socketwr)
                end
             end
          else
             if torch.typename(object) and torch.typename(object):find('torch.*Storage') then
                -- raw transfer ot storage
                object.zmq.send(object, process.socketwr)
             else
                -- serialize data first
                local f = torch.MemoryFile()
                f:binary()
                f:writeObject(object)
                local s = f:storage()
                -- then transmit raw storage
                send(process, s)
                f:close()
             end
          end
       end
parallel.send = send
--------------------------------------------------------------------------------
-- receive data
--------------------------------------------------------------------------------
receive = function(process, object, flags)
          -- parallel.print('IN REVEIVE FUNCTION')
          -- print(process)
             if object and type(object) == 'string' and object == 'noblock' or flags == 'noblock' then
                flags = parallel.zmq.NOBLOCK
             end
             local ret = true
             if not process.id then 
                -- receive all objects
                if object and object[1] and torch.typename(object[1]) and torch.typename(object[1]):find('torch.*Storage') then
                   -- user objects are storages, just fill them
                   local objects = object
                   for i,proc in pairs(process) do
                      if type(proc) == 'table' then
                         ret = object[i].zmq.recv(object[i], proc.socketrd, flags)
                      end
                   end
                else
                   -- receive raw storages
                   local storages = {}
                   for i,proc in pairs(process) do
                      if type(proc) == 'table' then
                         storages[i] = torch.CharStorage()
                         ret = storages[i].zmq.recv(storages[i], proc.socketrd, flags)
                      end
                   end
                   -- then un-serialize data objects
                   object = object or {}
                   for i,proc in pairs(process) do
                      if type(proc) == 'table' then
                         local f = torch.MemoryFile(storages[i])
                         f:binary()
                         object[i] = f:readObject()
                         f:close()
                      end
                   end
                end
             else
       
                if object and torch.typename(object) and torch.typename(object):find('torch.*Storage') then
                   -- raw receive of storage
          
                   ret = object.zmq.recv(object, process.socketrd, flags)
                else
                  -- print('else else part')
                   -- first receive raw storage
                   local s = torch.CharStorage()
                   _,ret = receive(process, s, flags)
                   if not ret then return end
                   -- then un-serialize data object
                   local f = torch.MemoryFile(s)
                   f:binary()
                   object = f:readObject()
                   f:close()
                end
             end
             return object, ret
          end
parallel.receive = receive
--------------------------------------------------------------------------------
-- close = clean up sockets
--------------------------------------------------------------------------------
close = function()
           print('closing session')
           if parallel.parent.id ~= -1 then
              sys.execute("sleep 1")
           end
           for _,process in pairs(parallel.children) do
              -- this is a bit brutal, but at least ensures that
              -- all forked children are *really* killed
              if type(process) == 'table' then
                 sys.execute('kill -9 ' .. process.unixid)
              end
           end
           if remotes then
              sys.execute("sleep 1")
              for _,remote in ipairs(remotes) do
                 -- second brutal thing: check for remote processes that
                 -- might have become orphans, and kill them
                 local prot = remote.protocol or 'ssh -Y'
                 local orphans = sys.fexecute(prot .. " " .. remote.ip .. " " ..
                                             "ps -ef | grep '" .. (rlua or 'luajit') .. "' "  ..
                                             "| awk '{if ($3 == 1) {print $2}}'")
                 local kill = 'kill -9 '
                 local pids = ''
                 for orphan in orphans:gfind('%d+') do
                    pids = pids .. orphan .. ' '
                 end
                 if pids ~= '' then
                    sys.execute(prot .. ' ' .. remote.ip .. ' "' .. kill .. pids .. '"')
                 end
              end
           end
        end

parallel.close = close
--------------------------------------------------------------------------------
-- all processes should use this print method
--------------------------------------------------------------------------------
local _print = glob.print
print = function(...)
   _print('<parallel#' .. glob.string.format('%03d', parallel.id) .. '>', ...)
        end
parallel.print = print
--------------------------------------------------------------------------------
-- add remote machine
-- the table given is a table with N entries, each entry being:
-- entry = {ip='IP_ADDR', protocol='PROTOCOL', lua='REMOTE_LUA_CMD_LINE', cores='NB_CORES'}
--------------------------------------------------------------------------------
addremote = function(...)
               local args = {...}
               local config
               if type(args[1]) == 'table' then 
                  config = args
                  if type(config[1][1]) == 'table' then config = config[1] end
               else 
                  config = {args} 
               end
               remotes = remotes or {cores=0}
               for i,entry in ipairs(config) do
                  glob.table.insert(remotes, entry)
                  remotes.cores = remotes.cores + entry.cores
               end
            end
parallel.addremote = addremote
--------------------------------------------------------------------------------
-- calibrate remote machines: this function executes as many processes a
-- cores declared for each machine, and assigns a speed coefficient to each
-- machine. Processes forked on these machines will inherit from this 
-- coefficient.
-- coefs: 1.0 is the fastest machine
--        0.5 means that the machine is 2x slower than the faster
-- so, typically, if a process has coef 0.5, you want to give it twice as less
-- stuff to process than the process that has coef 1.0.
--------------------------------------------------------------------------------
calibrate = function()
               -- only calibrate if 'remotes' have been declared
               if not remotes then
                  error('<parallel.calibrate> calibration can only be done after addremote() is called')
                  return
               end
               print('calibrating remote machines')
               -- calibration code:
               local calib = [[
                     require 'torch'
                     require 'sys'
                     s = torch.Tensor(10000):fill(1)
                     d = torch.Tensor(10000):fill(0)
                     parallel.yield()
                     sys.tic()
                     for i = 1,100000 do
                        d:add(13,s)
                     end
                     time = sys.toc()
                     parallel.parent:send(time)
               ]]
               -- run calibration on as many cores as available
               local forked = sfork(remotes.cores)
               forked:exec(calib)
               forked:join()
               local times = forked:receive()
               forked:sync()
               -- normalize times
               local max = 0
               local speed
               for i,time in pairs(times) do speed = 1/time if speed > max then max = speed end end
               for i,time in pairs(times) do times[i] = (1/time)/max end
               -- store coefs in each remote
               for _,remote in ipairs(remotes) do
                  for i,time in pairs(times) do
                     if forked[i].ip == remote.ip then
                        remote.speed = glob.math.floor( time*10 + 0.5 ) / 10
                        break
                     end
                  end
               end
            end
parallel.calibrate = calibrate
--------------------------------------------------------------------------------
-- create new process table, with methods
--------------------------------------------------------------------------------
_fill = function(process)
           process.join = join
           process.sync = sync
           process.send = send
           process.receive = receive
           process.exec = exec
        end
parallel._fill = _fill
--------------------------------------------------------------------------------
-- reset = forget all children, go back to initial state
-- TODO: this is the right place to properly terminate children
--------------------------------------------------------------------------------
reset = function()
           parallel.id = assignedid or 0
           parallel.ip = assignedip or "127.0.0.1"
           parallel.parent = assignedparent or {id = -1}
           parallel.children = {}
           parallel.processid = 1
           if parallel.parent.id ~= -1 then
              parallel.parent.receive = receive
              parallel.parent.send = send
           end
           _fill(parallel.children)
           parallel.nchildren = 0
           autoip()
        end
reset()
parallel.reset = reset

return parallel
