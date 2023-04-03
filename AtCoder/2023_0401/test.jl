function Altn()
    N = parse(Int, readline())
    S = readline()
    for i in 1:N-1
        if(S[i]==S[i+1])
            println("No")
            return nothing
        end
    end
    println("Yes")
end

function Chess()
    for i in 1:8
        S = readline()
        for j in 1:8
            if(S[j]=='*')
                ss = 'a' + j -1
                println("$(ss)$(9-i)")
                return nothing
            end
        end
    end
end

function deep(i::Int, sum::Int)

end
function GapE()
    N, X = parse.(Int, split(readline()))
    A = parse.(Int, split(readline()))
    sort!(A)
    for i in 1:N
        B = X + A[i]
        S = round(Int, log2(N))
        st = 1
        ed = N
        for j in 1:S
            if((ed-st)<4)
                break;
            else
                n = round(Int,(st+ed)/2)
                if(A[n]>B)
                    ed = n
                else
                    st = n
                end
            end            
        end
        for j in st:ed
            if(A[j]==B)
                println("Yes")
                return nothing
            end
        end
    end
    println("No")
end

function DM()
    N, M = parse.(Int, split(readline()))
    if(M>N^2)
        println(-1)
        return nothing
    else
        st = round(Int,M/N)
        if(st==0)
            st+=1
        end
        ed = round(Int, sqrt(M)+1)
        if(ed>N)
            ed = N
        end
        best = N^2
        for s in st:ed
            x = round(Int,M/s)
            if(x>N)
            else
                if(s*x<M)
                    x +=1
                end 
                if(best-s*x>0)
                    if(s*x>=M)
                        best = s*x
                    end
                end
            end
        end
        println(best)
    end
end

DM()