-- required libs
require 'parallel'

-- define code for workers:
function worker()
   -- a worker starts with a blank stack, we need to reload
   -- our libraries
   require 'sys'
   require 'torch'

   -- print from worker:
   parallel.print('Im a worker, my ID is: ' .. parallel.id .. ' and my IP: ' .. parallel.ip)

   -- define a storage to receive data from top process
   while true do
      -- yield = allow parent to terminate me
      m = parallel.yield()
      if m == 'break' then break end   

      -- receive data
      local t = parallel.parent:receive()
      io.write('.') io.flush()
      collectgarbage()

      parallel.print('received object with norm: ', t.data:norm())
      


      -- send some data back
      parallel.parent:send('this is my response')
   end
end

-- define code for parent:
function parent()
   -- print from top process
   parallel.print('Im the parent, my ID is: ' .. parallel.id)

   -- fork N processes
   parallel.nfork(4)

   -- parallel.addremote(...)
   -- parallel.calibrate()
   -- forked = parallel.sfork(parallel.remotes.cores)  -- fork as many processes as cores available
   -- for _,forked in ipairs(forked) do
   --    parallel.print('id: ' .. forked.id .. ', speed = ' .. forked.speed)
   -- end

   -- exec worker code in each process
   parallel.children:exec(worker)

   -- create a complex object to send to workers
   t = {name='my variable', data=torch.randn(100,100)}

   -- transmit object to each worker
   parallel.print('transmitting object with norm: ', t.data:norm())
   for i = 1,100 do
      parallel.children:join()
      parallel.children:send(t)
      -- print(parallel.children[1]:receive())
      replies = parallel.children:receive()
      print(replies)
   end
   parallel.print('transmitted data to all children')

   -- sync/terminate when all workers are done
   parallel.children:join('break')
   parallel.print('all processes terminated')
end

-- protected execution:
ok,err = pcall(parent)
if not ok then print(err) parallel.close() end