-- -- define code for workers:
function worker()
    require 'sys'
   require 'torch'
   parallel.print(os.execute("cd Singularity/seq2seq-elements/"))
     parallel.print('Im a worker, my ID is: ' .. parallel.id .. ' and my IP: ' .. parallel.ip)
     parallel.print('parallel.parent ', parallel.parent)
    parallel.print('HELLOTHERE')
    funcs = loadfile("functions.lua")
    funcs()

    --load in functions
    datafun = loadfile("data.lua")
    data = datafun()

    -- print from worker:
   

    first = true

   -- define a storage to receive data from top process
    while true do
        -- yield = allow parent to terminate me
        m = parallel.yield()
        if m == 'break' then break end   

        -- receive data
        local pkg = parallel.parent:receive()
        if first then 
            parallel.print('Recieved initialization parameters')
            cmd = pkg.cmd
            arg = pkg.arg
            
            train_data, valid_data, model, criterion, opt = main()

            first = false

            -- send some data back
            parallel.parent:send('Received parameters and loaded data successfully')
        else
            

            io.write('.') io.flush()
            collectgarbage()

            parallel.print('received object with index: ', pkg.index)


            for i = 1, #model.params do
                model.params[i]:copy(pkg.parameters[i])
            end

            local pkg_o = train_ind(pkg.index, model, criterion, train_data)


            -- send some data back

            parallel.print('sending back object with index: ', pkg.index)
            parallel.parent:send(pkg_o)
        end
    end
end

-- worker = [[
--       -- a worker starts with a blank stack, we need to reload
--       -- our libraries
--       require 'sys'
--       require 'torch'

--       -- print from worker:
--       parallel.print('Im a worker, my ID is: ' .. parallel.id .. ' and my IP: ' .. parallel.ip)

--       -- define a storage to receive data from top process
--       while true do
--          -- yield = allow parent to terminate me
--          m = parallel.yield()
--          if m == 'break' then break end

--          -- receive data
--          local t = parallel.parent:receive()
--          parallel.print('received object with norm: ', t.data:norm())

--          -- send some data back
--          parallel.parent:send('this is my response')
--       end
-- ]]
-- define code for parent:
function parent()
    require "package"
    

    

    local instances = { ['104.154.22.185'] ='mikes-instance-group-4phn',   
                        ['104.197.9.84'] = 'mikes-instance-group-8mir', 
                        ['104.154.82.175'] = 'mikes-instance-group-9dze', 
                        ['104.197.9.244'] = 'mikes-instance-group-de2i', 
                        ['146.148.102.75'] = 'mikes-instance-group-m6dp', 
                        ['104.197.179.152'] = 'mikes-instance-group-qwur', 
                        ['104.197.244.249'] = 'mikes-instance-group-uh3b', 
                        ['199.223.233.216'] = 'mikes-instance-group-usjf', 
                        ['104.197.44.175'] = 'mikes-instance-group-vlvn' }


    -- print from top process
    parallel.print('Im the parent, my ID is: ' .. parallel.id)

    -- local p = require 'posix'

    -- local pids = {}

    -- for k, v in pairs(instances) do
    --     local cpid = p.fork()
    --     if cpid == 0 then -- child reads from pipe

    --         setupEnvironment(v)
    --         p._exit(0)
    --     else -- parent writes to pipe
    --         table.insert(pids, cpid)
    --         -- wait for child to finish
            
    --     end
        
    -- end

    -- for k, v in pairs(pids) do

    --     p.wait(v)
    -- end
    

    -- for k, v in pairs(instances) do
    --     setupEnvironment(v)
    -- end

    train_data, valid_data, model, criterion, opt = main()

    n_proc= 1
    -- parallel.ip = "XX.XX.XX.XX"

    -- fork N processes
    -- parallel.nfork(n_proc)


    -- parallel.print('adding remote')


    old_path = package.path
    old_cpath = package.cpath

   

    -- -- parallel.print('ALERT', old_path)
    -- parallel.print('ALERT', package.path)

     package.path = "/home/michaelfarrell/.luarocks/share/lua/5.1/?.lua;/home/michaelfarrell/.luarocks/share/lua/5.1/?/init.lua;/home/michaelfarrell/torch/install/share/lua/5.1/?.lua;/home/michaelfarrell/torch/install/share/lua/5.1/?/init.lua;./?.lua;/home/michaelfarrell/Singularity/seq2seq-elements/?.lua;/home/michaelfarrell/torch/install/share/luajit-2.1.0-beta1/?.lua;/usr/local/share/lua/5.1/?.lua;/usr/local/share/lua/5.1/?/init.lua"
    package.cpath = "/home/michaelfarrell/.luarocks/lib/lua/5.1/?.so;/home/michaelfarrell/torch/install/lib/lua/5.1/?.so;./?.so;/usr/local/lib/lua/5.1/?.so;/usr/local/lib/lua/5.1/loadall.so"

    parallel.addremote( {ip='mikes-instance-group-4phn', cores=4, lua='/home/michaelfarrell/torch/install/bin/th', protocol="gcloud compute ssh"})--'ssh -ttq -i ~/.ssh/my-ssh-key'})   --,
    
    -- michaelfarrell@104.197.157.136

    
   -- package.path = "/home/michaelfarrell/.luarocks/share/lua/5.1/?.lua;/home/michaelfarrell/.luarocks/share/lua/5.1/?/init.lua;/home/michaelfarrell/torch/install/share/lua/5.1/?.lua;/home/michaelfarrell/torch/install/share/lua/5.1/?/init.lua;./?.lua;/home/michaelfarrell/Singularity/seq2seq-elements/?.lua;/home/michaelfarrell/torch/install/share/luajit-2.1.0-beta1/?.lua;/usr/local/share/lua/5.1/?.lua;/usr/local/share/lua/5.1/?/init.lua"
   --  package.cpath = "/home/michaelfarrell/.luarocks/lib/lua/5.1/?.so;/home/michaelfarrell/torch/install/lib/lua/5.1/?.so;./?.so;/usr/local/lib/lua/5.1/?.so;/usr/local/lib/lua/5.1/loadall.so"

 --      package.path = "/Users/candokevin/.luarocks/share/lua/5.1/?.lua;/Users/candokevin/.luarocks/share/lua/5.1/?/init.lua;/Users/candokevin/torch/install/share/lua/5.1/?.lua;/Users/candokevin/torch/install/share/lua/5.1/?/init.lua;./?.lua;/Users/candokevin/torch/install/share/luajit-2.1.0-beta1/?.lua;/usr/local/share/lua/5.1/?.lua;/usr/local/share/lua/5.1/?/init.lua"
 -- package.cpath = " /Users/candokevin/.luarocks/lib/lua/5.1/?.so;/Users/candokevin/torch/install/lib/lua/5.1/?.so;/Users/candokevin/torch/install/lib/?.dylib;./?.so;/usr/local/lib/lua/5.1/?.so;/usr/local/lib/lua/5.1/loadall.so"
    
 --    parallel.addremote({ip='candokevin@10.251.57.175', cores=4, lua='/Users/candokevin/torch/install/bin/th', protocol='ssh -ttq'})

    -- parallel.addremote({ip='michaelfarrell@10.251.54.86', cores=4, lua='/Users/michaelfarrell/torch/install/bin/th', protocol='ssh -ttq'})

    -- -- parallel.addremote( {ip='michaelfarrell@104.197.157.136', cores=8, lua='~/torch/install/bin/th', protocol='ssh -ttq -i ~/.ssh/my-ssh-key'},
    -- --                     {ip='michaelfarrell@104.197.111.94', cores=8, lua='~/torch/install/bin/th', protocol='ssh -ttq -i ~/.ssh/my-ssh-key'}) --,
    --                     -- {ip='michaelfarrell@199.223.233.216', cores=8, lua='/home/michaelfarrell/torch/install/bin/th', protocol='ssh -i ~/.ssh/my-ssh-key'},
    --                     -- {ip='michaelfarrell@104.197.143.177', cores=8, lua='/home/michaelfarrell/torch/install/bin/th', protocol='ssh -i ~/.ssh/my-ssh-key'},
    --                     -- {ip='michaelfarrell@104.197.179.152', cores=8, lua='/home/michaelfarrell/torch/install/bin/th', protocol='ssh -i ~/.ssh/my-ssh-key'},
    --                     -- {ip='michaelfarrell@104.197.9.84', cores=8, lua='/home/michaelfarrell/torch/install/bin/th', protocol='ssh -i ~/.ssh/my-ssh-key'},
    --                     -- {ip='michaelfarrell@104.154.16.196', cores=8, lua='/home/michaelfarrell/torch/install/bin/th', protocol='ssh -i ~/.ssh/my-ssh-key'},
    --                     -- {ip='michaelfarrell@104.154.82.175', cores=8, lua='/home/michaelfarrell/torch/install/bin/th', protocol='ssh -i ~/.ssh/my-ssh-key'},
    --                     -- {ip='michaelfarrell@104.197.9.244', cores=8, lua='/home/michaelfarrell/torch/install/bin/th', protocol='ssh -i ~/.ssh/my-ssh-key'})


    -- -- parallel.addremote( {ip='mikes-instance-group-4phn', cores=8, lua=paths.findprogram(bin_name), protocol='gcloud compute ssh'},
    -- --                     {ip='mikes-instance-group-8mir', cores=8, lua=paths.findprogram(bin_name), protocol='gcloud compute ssh'},
    -- --                     {ip='mikes-instance-group-9dze', cores=8, lua=paths.findprogram(bin_name), protocol='gcloud compute ssh'},
    -- --                     {ip='mikes-instance-group-de2i', cores=8, lua=paths.findprogram(bin_name), protocol='gcloud compute ssh'},
    -- --                     {ip='mikes-instance-group-m6dp', cores=8, lua=paths.findprogram(bin_name), protocol='gcloud compute ssh'},
    -- --                     {ip='mikes-instance-group-qwur', cores=8, lua=paths.findprogram(bin_name), protocol='gcloud compute ssh'},
    -- --                     {ip='mikes-instance-group-uh3b', cores=8, lua=paths.findprogram(bin_name), protocol='gcloud compute ssh'},
    -- --                     {ip='mikes-instance-group-usjf', cores=8, lua=paths.findprogram(bin_name), protocol='gcloud compute ssh'},
    -- --                     {ip='mikes-instance-group-vlvn', cores=8, lua=paths.findprogram(bin_name), protocol='gcloud compute ssh'})

    
    -- -- parallel.calibrate()
    -- parallel.print('Forking')
    -- parallel.print(n_proc)
    -- package.cpath = '/home/michaelfarrell/torch/install/lib/lua/5.1/?.so;' ..package.cpath
    -- -- package.path = '/home/michaelfarrell/torch/install/share/lua/5.1/?/init.lua;' .. package.path
    -- package.path = '/home/michaelfarrell/lua---?/init.lua;' .. package.path
    
    parallel.sfork(n_proc)
    -- parallel.print(parallel.nchildren)
    -- forked = parallel.sfork(parallel.remotes.cores)
    parallel.print('Forked')
    parallel.print('parallel.id ', parallel.id)
    parallel.print('parallel.parent ', parallel.parent)

    package.path = old_path
    package.cpath = old_cpath


    -- exec worker code in each process
    parallel.children:exec(worker)

    parallel.print('post worker farm')

    --send the global parameters to the children
    parallel.children:join()
    parallel.children:send({cmd = cmd, arg = arg})
    replies = parallel.children:receive()
    parallel.print(replies)

    --trainmodel
    train(model, criterion, train_data, valid_data)


    parallel.print('transmitted data to all children')

    -- sync/terminate when all workers are done
    parallel.children:join('break')
    parallel.print('all processes terminated')
end

function setupEnvironment(instance)

     -- print from worker:
    parallel.print('=====>Setting up environent for ' .. instance)

    parallel.print('Ensure that the startup script has been run')
    os.execute('gcloud compute copy-files ../../startup.sh ' .. instance..':~/')
    os.execute('echo "bash startup.sh" | gcloud compute ssh ' .. instance)

    parallel.print('Ensure that the MovieTriples have been copied over')
    os.execute('(echo "ls Singularity/data" | gcloud compute ssh '.. instance .. ' | grep -q MovieTriples) || gcloud compute copy-files ../data/MovieTriples '.. instance .. ':~/Singularity/data')
    
    parallel.print('Ensure that preprocessing script has been run')
    os.execute('(echo "ls Singularity/seq2seq-elements/data" | gcloud compute ssh '.. instance .. ' | grep -q train_src_words.txt) || (echo "python Singularity/seq2seq-elements/preprocess.py --data_directory Singularity/data/MovieTriples/ --output_directory Singularity/seq2seq-elements/data/" | gcloud compute ssh '.. instance .. ')')

end

