-- Worker code
function worker()
    require "package"

    -- Alert successfully started up
    parallel.print('Im a worker, my ID is: ',  parallel.id, ' and my IP: ', parallel.ip)
    parallel.print('parallel.parent ', parallel.parent)

    -- Global indicating is a child
    ischild = true


    -- Number of packages received
    local n_pkg = 0

    ext = ""

    while true do
        -- Allow the parent to terminate the child
        m = parallel.yield()
        if m == 'break' then break end   

        -- Receive data
        local pkg = parallel.parent:receive()
        if n_pkg == 0 then 
            -- This is the first time receiving a package, it has the globals

            parallel.print('Recieved initialization parameters')
            cmd, arg, ext = pkg.cmd, pkg.arg, pkg.ext

            -- Load in functions
            funcs = loadfile(ext .. "model_functions.lua")
            funcs()

            -- Load in data
            datafun = loadfile(ext .. "data.lua")
            data = datafun()
            
            -- Load in data to client
            train_data, valid_data, model, criterion, opt = main()


            --point the wordvec to the right place
            opt.pre_word_vecs = opt.extension .. opt.pre_word_vecs

            first = false

            -- send some data back
            parallel.parent:send('Received parameters and loaded data successfully')
        else
            -- Make sure to clean everything up since big files are being passed
            io.write('.') io.flush()
            collectgarbage()

            parallel.print('received params from batch with index: ', pkg.index)

            -- Load in the parameters sent from the parent
            for i = 1, #model.params do
                model.params[i]:copy(pkg.parameters[i])
            end
            parallel.print('a')

            -- Training the model at the given index
            local pkg_o = train_ind(pkg.index, model, criterion, train_data)

             parallel.print('z')


            -- send some data back
            parallel.print('sending back derivative for batch with index: ', pkg.index)
            parallel.parent:send(pkg_o)
        end
        n_pkg = n_pkg + 1
    end
end

function setup_servers()
    local p = require 'posix'

    local instances = { ['104.154.22.185'] ='mikes-instance-group-4phn',   
                        ['104.197.9.84'] = 'mikes-instance-group-8mir', 
                        ['104.154.82.175'] = 'mikes-instance-group-9dze', 
                        ['104.197.9.244'] = 'mikes-instance-group-de2i', 
                        ['146.148.102.75'] = 'mikes-instance-group-m6dp', 
                        ['104.197.179.152'] = 'mikes-instance-group-qwur', 
                        ['104.197.244.249'] = 'mikes-instance-group-uh3b', 
                        ['199.223.233.216'] = 'mikes-instance-group-usjf', 
                        ['104.197.44.175'] = 'mikes-instance-group-vlvn' }

    local pids = {}
    for k, v in pairs(instances) do
        local cpid = p.fork()
        if cpid == 0 then -- child reads from pipe

            setupEnvironment(v)
            p._exit(0)
        else -- parent writes to pipe
            table.insert(pids, cpid)
            -- wait for child to finish
            
        end
        
    end

    for k, v in pairs(pids) do

        p.wait(v)
    end
    

    for k, v in pairs(instances) do
        setupEnvironment(v)
    end

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

-- The parent process function
function parent()
    require "package"

    -- Print from parent process
    parallel.print('Im the parent, my ID is: ',  parallel.id)

    if opt.setup_servers then
        parallel.print('Setting up remote servers')
        setup_servers()
    end

    parallel.print('Loading data, parameters, model...')
    ext = ""
    train_data, valid_data, model, criterion, opt = main()

    old_path = package.path
    old_cpath = package.cpath


    if opt.remote then
        parallel.print('Runnign remotely')
        -- Setup remote servers this isnt working i was playing around with the path variables and stuff but couldnt get it to connect
        -- most likely either a problem with the google server not letting me in or im not setting up the lua environment correctly
        

      
        -- package.cpath = '/home/michaelfarrell/torch/install/lib/lua/5.1/?.so;' ..package.cpath
        -- package.path = '/home/michaelfarrell/torch/install/share/lua/5.1/?/init.lua;' .. package.path
        -- package.path = '/home/michaelfarrell/lua---?/init.lua;' .. package.path

        package.path = "/home/michaelfarrell/.luarocks/share/lua/5.1/?.lua;/home/michaelfarrell/.luarocks/share/lua/5.1/?/init.lua;/home/michaelfarrell/torch/install/share/lua/5.1/?.lua;/home/michaelfarrell/torch/install/share/lua/5.1/?/init.lua;./?.lua;/home/michaelfarrell/Singularity/seq2seq-elements/?.lua;/home/michaelfarrell/torch/install/share/luajit-2.1.0-beta1/?.lua;/usr/local/share/lua/5.1/?.lua;/usr/local/share/lua/5.1/?/init.lua"
        package.cpath = "/home/michaelfarrell/.luarocks/lib/lua/5.1/?.so;/home/michaelfarrell/torch/install/lib/lua/5.1/?.so;./?.so;/usr/local/lib/lua/5.1/?.so;/usr/local/lib/lua/5.1/loadall.so"

        -- parallel.addremote( {ip='mikes-instance-group-4phn', cores=4, lua='/home/michaelfarrell/torch/install/bin/th', protocol="gcloud compute ssh"})--'ssh -ttq -i ~/.ssh/my-ssh-key'})   --,
    
        parallel.addremote( {ip='michaelfarrell@104.197.111.94', cores=4, lua='/home/michaelfarrell/torch/install/bin/th', protocol='ssh -o "StrictHostKeyChecking no" -i ~/.ssh/gcloud-sshkey'})
    
        -- parallel.addremote({ip='candokevin@10.251.57.175', cores=4, lua='/Users/candokevin/torch/install/bin/th', protocol='ssh -ttq'})

        -- parallel.addremote({ip='michaelfarrell@10.251.54.86', cores=4, lua='/Users/michaelfarrell/torch/install/bin/th', protocol='ssh -ttq'})

    
        -- parallel.calibrate()
    elseif opt.localhost then
        parallel.addremote({ip='localhost', cores=4, lua=opt.torch_path, protocol='ssh -ttq'})
        -- parallel.addremote({ip='michaelfarrell@10.251.50.115', cores=4, lua=opt.torch_path, protocol='ssh -ttq'})
    elseif opt.kevin then        
        package.path = "/Users/candokevin/.luarocks/share/lua/5.1/?.lua;/Users/candokevin/.luarocks/share/lua/5.1/?/init.lua;/Users/candokevin/torch/install/share/lua/5.1/?.lua;/Users/candokevin/torch/install/share/lua/5.1/?/init.lua;./?.lua;/Users/candokevin/torch/install/share/luajit-2.1.0-beta1/?.lua;/usr/local/share/lua/5.1/?.lua;/usr/local/share/lua/5.1/?/init.lua"
        package.cpath = " /Users/candokevin/.luarocks/lib/lua/5.1/?.so;/Users/candokevin/torch/install/lib/lua/5.1/?.so;/Users/candokevin/torch/install/lib/?.dylib;./?.so;/usr/local/lib/lua/5.1/?.so;/usr/local/lib/lua/5.1/loadall.so"
        
        parallel.addremote({ip='candokevin@10.251.53.101', cores=4, lua='/Users/candokevin/torch/install/bin/th', protocol='ssh -ttq'})
        

    end
    
    parallel.print('Forking ', opt.n_proc, ' processes')
    parallel.sfork(opt.n_proc)
 
    parallel.print('Forked')
    parallel.print('parallel.id ', parallel.id)
    parallel.print('parallel.parent ', parallel.parent)

    -- Set path back
    package.path = old_path
    package.cpath = old_cpath


    -- exec worker code in each process
    parallel.children:exec(worker)
    parallel.print('Finished telling workers to execute')

    --send the global parameters to the children
    parallel.children:join()
    parallel.print('Sending parameters to children')
    parallel.children:send({cmd = cmd, arg = arg, ext = opt.extension})

    -- Get the responses from the children
    replies = parallel.children:receive()
    parallel.print('Replies from children', replies)

    -- Train the model
    train(model, criterion, train_data, valid_data)
    parallel.print('Finished training the model')

    -- sync/terminate when all workers are done
    parallel.children:join('break')
    parallel.print('All processes terminated')
end