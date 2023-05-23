function Attack()
    A, B = parse.(Int, split(readline()))
    sw = A%B
    if(sw>0)
        println(div(A,B) + 1)
    else
        println(div(A,B))
    end
end

function FS()
    H, W = parse.(Int, split(readline()))
    S::Vector{String} = []
    for it in 1:H
        push!(S, readline())
    end
    dir =[(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1), (1,-1), (-1,1)]
    for i in 1:H
        id_s = findall(x->x=='s' ,S[i])
        for is in id_s
            for (x,y) in dir
                try
                    if(S[i+x][is+y]=='n' && S[i+2x][is+2y]=='u'&& S[i+3x][is+3y]=='k' && S[i+4x][is+4y]=='e')
                        for d in 0:4
                            println("$(i+d*x) $(is+d*y)")
                        end
                        return nothing
                    end
                catch
                end
            end
        end
    end
    return nothing
end

function AE()
    N, M = parse.(Int, split(readline()))
    S::Vector{String} =[]
    for it in 1:N
        push!(S, readline())
    end
    child::Vector{Vector{Int}} = [[] for i in 1:N]
    for ns in 1:N
        for nn in 1:N
            if(ns==nn)
                continue
            else
                count = 0
                for l in 1:M
                    if(S[ns][l]!=S[nn][l])
                        count += 1
                    end
                    if(count>1)
                        break
                    end
                end
                if(count==1)
                    push!(child[ns], nn)
                end
            end
        end
    end
    ans = 0
    for it in 1:N
        if(length(child[it])==1)
            ans += 1
        elseif(length(child[it])==0)
            println("No")
            return nothing
        end
        if(ans>2)
            println("No")
            return nothing
        end
    end
    println("Yes")
end

function find_b(A, B, D, it_a, it_b)
    a = A[end-it_a]
    for it_bb in it_b:length(B)-1
        b = B[end-it_bb]
        if(abs(a-b)<=D)
            println(a+b)
            return -1
        elseif(b<a)
            return it_bb
        end
    end
    println(-1)
    return -100
end
function find_a(A, B, D, it_a, it_b)
    b = B[end-it_b]
    for it_aa in it_a:length(A)-1
        a = A[end-it_aa]
        if(abs(b-a)<=D)
            println(a+b)
            return -1
        elseif(a<b)
            return it_aa
        end
    end
    println(-1)
    return -100
end
function IG()
    N, M, D = parse.(Int, split(readline()))
    A = parse.(Int, split(readline()))
    B = parse.(Int, split(readline()))

    sort!(A)
    sort!(B)
    it_a = 0
    it_b = 0
    #println("start")
    while true
        it_b = find_b(A, B, D, it_a, it_b)
        if(it_b<0)
            return nothing
        end
        it_a = find_a(A, B, D, it_a, it_b)
        if(it_a<0)
            return nothing
        end
        #println("$(it_a), $(it_b)")
    end
end

IG()