function TC()
    N = parse(Int, readline())
    S = readline()
    st = 0
    ed = N
    as = 0
    for i in 1:N
        if(S[i]=='|')
            if(st == 0)
                st = i
            else
                ed =i
            end
        elseif(S[i]=='*')
            as = i
        end
    end
    if(st<as && as<ed)
        println("in")
    else
        println("out")
    end
end

function TT()
    N, T = parse.(Int, split(readline()))
    C = parse.(Int, split(readline()))
    R = parse.(Int, split(readline()))
    player = findall(x -> x==T, C)
    if(isempty(player))
        player = findall(x -> x==C[1], C)
    end
    pw::Vector{Int} = []
    for i in 1:size(player)[1]
        push!(pw, R[player[i]])
    end

    Win = findmax(pw)[2]
    println(player[Win])
end

function Dango()
    N = parse(Int, readline())
    S = readline()
    l = 0
    l_MAX = 0
    for i in 1:N
        if(S[i]=='o')
            l+=1
        else
            l=0
        end
        if(l>l_MAX)
            l_MAX = l
        end
    end
    if(l_MAX==0 || l_MAX==N)
        l_MAX = -1
    end
    println(l_MAX)
end

function FbQ()
    N = parse(Int, readline())
    #st = 1
    st_new = Int(round(N/2))
    st = 1
    ed = N
    count = 0
    for i in 1:20
        println("? $(st_new)")
        ans = parse(Int, readline())
        #st_old = st_new
        if(ans == 0)
            st = st_new
            st_new += Int(round(N/(2^(i+1))))
        else
            ed=st_new
            st_new -= Int(round(N/(2^(i+1))))
        end
        if(abs(st-ed)<20-i)
            break
        end
    end
    for i in 1:ed-st
        println("? $(st+i)")
        ans = parse(Int, readline())
        if(ans==1)
            println("! $(st+i-1)")
            return nothing
        end
        if(i==ed-st-1)
            println("! $(ed-1)")
            return nothing
        end
    end
end

#graph: Vec{Vec{}}の連結情報, start: 初期位置, dist:距離
function bfs(graph, start)
    dist = fill(Inf, length(graph))
    dist[start] = 0
    queue = [start]
    while !isempty(queue)
        node = queue[1]
        queue = queue[2:end]
        for neighbor in graph[node]
            if dist[neighbor] == Inf
                dist[neighbor] = dist[node] + 1
                push!(queue, neighbor)
            end
        end
    end
    return dist
end

function NBV()
    N, M = parse.(Int, split(readline()))
    v_mat = [[] for i in 1:N]
    for i in 1:M
        u, v = parse.(Int, split(readline()))
        push!(v_mat[u], v)
        push!(v_mat[v], u)
    end
    K = parse(Int, readline())
    if K == 0
        println("Yes")
        bw = ones(Int, N)
        for i in 1:N
            print(bw[i])
        end
        println("")
        return nothing
    end
    cond = [parse.(Int, split(readline())) for i in 1:K]
    sort!(cond, by = x -> x[2])

    bw = ones(Int, N)
    dist = zeros(Int, K, N)
    for i in 1:K
        dist[i,:] = bfs(v_mat, cond[i][1])
        for s in 1:N
            if(dist[i,s] < cond[i][2])
                bw[s] = 0
            end
        end
    end
    
    for i in 1:K
        cc = false
        for s in 1:N
            if(dist[i,s] == cond[i][2] && bw[s]>0)
                cc = true
            end
        end
        if(!cc) 
            println("No")
            return nothing
        end
    end
    println("Yes")
    for i in 1:N
        if(bw[i]==1)
            print(1)
        else
            print(0)
        end
    end
    println("")
end

function SS()
    S = readline()
    count = 0
    N = length(S)
    alp = 26

    sig = (N+1)*ones(Int, N, alp)#0~N-1
    for w in 1:alp
        ss = Char('a'+w-1)
        for s in 1:N
            ind = findall(ss, S)
            #prepend!(ind,0)
            for i in 1:size(ind)[1]
                if(i==1)
                    st = 1
                else
                    st = ind[i-1]
                end
                ed = ind[i]
                sig[st:min(ed,N),w] .= ed
            end
        end
    end

    dp::Vector{Matrix{Int}} = [zeros(Int, j-1, N-j+1) for j in 2:N]
    #println(dp[4])
    for j in 2:N
        cj = Int(S[j]-'a'+1)
        i = sig[1, cj]
        if (i<j)
            if(sig[1, cj]==i)
                #println("i:$(i), j:$(j)")
                count += 1
            end
            (dp[j-1])[i,1] = 1
            if(j==N) 
                continue
            end
            for ip in i+1:j-1
                cp = Int(S[ip]-'a'+1)
                for kp in j+1:N
                #for jj in j:N-1
                    if(S[ip]==S[kp])
                        #println("i:$(i), j:$(j), ip:$(ip), jp:$(jp), jj:$(jj)") 
                        #println(dp[j])
                        (dp[j-1])[ip, kp-j+1] += sum((dp[j-1])[i, 1:kp-j])
                        for k in j:kp-1
                            if(sig[ip+1,cj]==j && sig[k,cp]==kp)# 
                                count += (dp[j-1])[i, k-j+1]
                            end
                        end
                        
                    end
                end
            end
        end
    end
    println(count)
end

SS()

#0010011
#000000000000001