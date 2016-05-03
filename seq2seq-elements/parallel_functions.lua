-- define code for workers:
function worker()
    print(os.execute("pwd"))
    funcs = loadfile("functions.lua")
    funcs()

    --load in functions
    datafun = loadfile("data.lua")
    data = datafun()

    -- print from worker:
    parallel.print('Im a worker, my ID is: ' .. parallel.id .. ' and my IP: ' .. parallel.ip)

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


-- define code for parent:
function parent()

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

    -- -- fork N processes
    -- parallel.nfork(n_proc)


    parallel.print('adding remote')

    -- parallel.addremote( {ip='104.154.22.185', cores=8, lua='/home/michaelfarrell/torch', protocol='gcloud compute ssh mikes-instance-group-4phn'},
    --                     {ip='104.197.9.84', cores=8, lua='/home/michaelfarrell/torch', protocol='gcloud compute ssh mikes-instance-group-8mir'},
    --                     {ip='104.154.82.175', cores=8, lua='/home/michaelfarrell/torch', protocol='gcloud compute ssh mikes-instance-group-9dze'},
    --                     {ip='104.197.9.244', cores=8, lua='/home/michaelfarrell/torch', protocol='gcloud compute ssh mikes-instance-group-de2i'},
    --                     {ip='146.148.102.75', cores=8, lua='/home/michaelfarrell/torch', protocol='gcloud compute ssh mikes-instance-group-m6dp'},
    --                     {ip='104.197.179.152', cores=8, lua='/home/michaelfarrell/torch', protocol='gcloud compute ssh mikes-instance-group-qwur'},
    --                     {ip='104.197.244.249', cores=8, lua='/home/michaelfarrell/torch', protocol='gcloud compute ssh mikes-instance-group-uh3b'},
    --                     {ip='199.223.233.216', cores=8, lua='/home/michaelfarrell/torch', protocol='gcloud compute ssh mikes-instance-group-usjf'},
    --                     {ip='104.197.44.175', cores=8, lua='/home/michaelfarrell/torch', protocol='gcloud compute ssh mikes-instance-group-vlvn'})

    -- local bin_name = jit and 'luajit' or 'lua'
    -- parallel.addremote( {ip='mikes-instance-group-4phn', cores=8, lua='/home/michaelfarrell/torch', protocol='gcloud compute ssh'},
    --                     {ip='mikes-instance-group-8mir', cores=8, lua='/home/michaelfarrell/torch', protocol='gcloud compute ssh'},
    --                     {ip='mikes-instance-group-9dze', cores=8, lua='/home/michaelfarrell/torch', protocol='gcloud compute ssh'},
    --                     {ip='mikes-instance-group-de2i', cores=8, lua='/home/michaelfarrell/torch', protocol='gcloud compute ssh'},
    --                     {ip='mikes-instance-group-m6dp', cores=8, lua='/home/michaelfarrell/torch', protocol='gcloud compute ssh'},
    --                     {ip='mikes-instance-group-qwur', cores=8, lua='/home/michaelfarrell/torch', protocol='gcloud compute ssh'},
    --                     {ip='mikes-instance-group-uh3b', cores=8, lua='/home/michaelfarrell/torch', protocol='gcloud compute ssh'},
    --                     {ip='mikes-instance-group-usjf', cores=8, lua='/home/michaelfarrell/torch', protocol='gcloud compute ssh'},
    --                     {ip='mikes-instance-group-vlvn', cores=8, lua='/home/michaelfarrell/torch', protocol='gcloud compute ssh'})

-- '/home/michaelfarrell/torch/install/bin/torch-activate'

    -- parallel.addremote( {ip='104.154.22.185', cores=8, lua='/home/michaelfarrell/torch', protocol='ssh -Y -i ~/.ssh/my-ssh-key'},
    --                     {ip='104.197.9.84', cores=8, lua='/home/michaelfarrell/torch', protocol='ssh -Y -i ~/.ssh/my-ssh-key'},
    --                     {ip='104.154.82.175', cores=8, lua='/home/michaelfarrell/torch', protocol='ssh -Y -i ~/.ssh/my-ssh-key'},
    --                     {ip='104.197.9.244', cores=8, lua='/home/michaelfarrell/torch', protocol='ssh -Y -i ~/.ssh/my-ssh-key'},
    --                     {ip='146.148.102.75', cores=8, lua='/home/michaelfarrell/torch', protocol='ssh -Y -i ~/.ssh/my-ssh-key'},
    --                     {ip='104.197.179.152', cores=8, lua='/home/michaelfarrell/torch', protocol='ssh -Y -i ~/.ssh/my-ssh-key'},
    --                     {ip='104.197.244.249', cores=8, lua='/home/michaelfarrell/torch', protocol='ssh -Y -i ~/.ssh/my-ssh-key'},
    --                     {ip='199.223.233.216', cores=8, lua='/home/michaelfarrell/torch', protocol='ssh -Y -i ~/.ssh/my-ssh-key'},
    --                     {ip='104.197.44.175', cores=8, lua='/home/michaelfarrell/torch', protocol='ssh -Y -i ~/.ssh/my-ssh-key'})

    -- parallel.addremote( {ip='kyang01@104.154.22.185', cores=8, lua='/home/michaelfarrell/torch/install/bin/torch-activate', protocol='ssh -i ~/.ssh/my-ssh-key'},
    --                     {ip='kyang01@104.197.9.84', cores=8, lua='/home/michaelfarrell/torch/install/bin/torch-activate', protocol='ssh -i ~/.ssh/my-ssh-key'},
    --                     {ip='kyang01@104.154.82.175', cores=8, lua='/home/michaelfarrell/torch/install/bin/torch-activate', protocol='ssh -i ~/.ssh/my-ssh-key'},
    --                     {ip='kyang01@104.197.9.244', cores=8, lua='/home/michaelfarrell/torch/install/bin/torch-activate', protocol='ssh -i ~/.ssh/my-ssh-key'},
    --                     {ip='kyang01@146.148.102.75', cores=8, lua='/home/michaelfarrell/torch/install/bin/torch-activate', protocol='ssh -i ~/.ssh/my-ssh-key'},
    --                     {ip='kyang01@104.197.179.152', cores=8, lua='/home/michaelfarrell/torch/install/bin/torch-activate', protocol='ssh -i ~/.ssh/my-ssh-key'},
    --                     {ip='kyang01@104.197.244.249', cores=8, lua='/home/michaelfarrell/torch/install/bin/torch-activate', protocol='ssh -i ~/.ssh/my-ssh-key'},
    --                     {ip='kyang01@199.223.233.216', cores=8, lua='/home/michaelfarrell/torch/install/bin/torch-activate', protocol='ssh -i ~/.ssh/my-ssh-key'},
    --                     {ip='kyang01@104.197.44.175', cores=8, lua='/home/michaelfarrell/torch/install/bin/torch-activate', protocol='ssh -i ~/.ssh/my-ssh-key'})
    parallel.addremote( {ip='michaelfarrell@104.197.157.136 "bash startup.sh; cd Singularity/seq2seq-elements/; source ~/.profile; source ~/.bashrc; (echo require [[env]] > testf.lua); th testf.lua; echo hi;  pwd; bash"', cores=8, lua='/home/michaelfarrell/torch/install/bin/th', protocol='ssh -ttq -i ~/.ssh/my-ssh-key'})   --,

                        -- {ip='localhost', cores=8, lua='/Users/michaelfarrell/torch/install/bin/th'})

    -- parallel.addremote( {ip='michaelfarrell@104.197.157.136', cores=8, lua='~/torch/install/bin/th', protocol='ssh -ttq -i ~/.ssh/my-ssh-key'},
    --                     {ip='michaelfarrell@104.197.111.94', cores=8, lua='~/torch/install/bin/th', protocol='ssh -ttq -i ~/.ssh/my-ssh-key'}) --,
                        -- {ip='michaelfarrell@199.223.233.216', cores=8, lua='/home/michaelfarrell/torch/install/bin/th', protocol='ssh -i ~/.ssh/my-ssh-key'},
                        -- {ip='michaelfarrell@104.197.143.177', cores=8, lua='/home/michaelfarrell/torch/install/bin/th', protocol='ssh -i ~/.ssh/my-ssh-key'},
                        -- {ip='michaelfarrell@104.197.179.152', cores=8, lua='/home/michaelfarrell/torch/install/bin/th', protocol='ssh -i ~/.ssh/my-ssh-key'},
                        -- {ip='michaelfarrell@104.197.9.84', cores=8, lua='/home/michaelfarrell/torch/install/bin/th', protocol='ssh -i ~/.ssh/my-ssh-key'},
                        -- {ip='michaelfarrell@104.154.16.196', cores=8, lua='/home/michaelfarrell/torch/install/bin/th', protocol='ssh -i ~/.ssh/my-ssh-key'},
                        -- {ip='michaelfarrell@104.154.82.175', cores=8, lua='/home/michaelfarrell/torch/install/bin/th', protocol='ssh -i ~/.ssh/my-ssh-key'},
                        -- {ip='michaelfarrell@104.197.9.244', cores=8, lua='/home/michaelfarrell/torch/install/bin/th', protocol='ssh -i ~/.ssh/my-ssh-key'})

      -- parallel.addremote( {ip='mikes-instance-group-4phn', cores=8, protocol='gcloud compute ssh'},
      --                   {ip='mikes-instance-group-8mir', cores=8, protocol='gcloud compute ssh'},
      --                   {ip='mikes-instance-group-9dze', cores=8,  protocol='gcloud compute ssh'},
      --                   {ip='mikes-instance-group-de2i', cores=8,  protocol='gcloud compute ssh'},
      --                   {ip='mikes-instance-group-m6dp', cores=8,  protocol='gcloud compute ssh'},
      --                   {ip='mikes-instance-group-qwur', cores=8,  protocol='gcloud compute ssh'},
      --                   {ip='mikes-instance-group-uh3b', cores=8,  protocol='gcloud compute ssh'},
      --                   {ip='mikes-instance-group-usjf', cores=8,  protocol='gcloud compute ssh'},
      --                   {ip='mikes-instance-group-vlvn', cores=8,  protocol='gcloud compute ssh'})

    -- parallel.addremote( {ip='mikes-instance-group-4phn', cores=8, lua=paths.findprogram(bin_name), protocol='gcloud compute ssh'},
    --                     {ip='mikes-instance-group-8mir', cores=8, lua=paths.findprogram(bin_name), protocol='gcloud compute ssh'},
    --                     {ip='mikes-instance-group-9dze', cores=8, lua=paths.findprogram(bin_name), protocol='gcloud compute ssh'},
    --                     {ip='mikes-instance-group-de2i', cores=8, lua=paths.findprogram(bin_name), protocol='gcloud compute ssh'},
    --                     {ip='mikes-instance-group-m6dp', cores=8, lua=paths.findprogram(bin_name), protocol='gcloud compute ssh'},
    --                     {ip='mikes-instance-group-qwur', cores=8, lua=paths.findprogram(bin_name), protocol='gcloud compute ssh'},
    --                     {ip='mikes-instance-group-uh3b', cores=8, lua=paths.findprogram(bin_name), protocol='gcloud compute ssh'},
    --                     {ip='mikes-instance-group-usjf', cores=8, lua=paths.findprogram(bin_name), protocol='gcloud compute ssh'},
    --                     {ip='mikes-instance-group-vlvn', cores=8, lua=paths.findprogram(bin_name), protocol='gcloud compute ssh'})

    
    -- parallel.calibrate()
    parallel.print('Forking')
    forked = parallel.sfork(n_proc)
    -- forked = parallel.sfork(parallel.remotes.cores)
    parallel.print('Forked')

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

