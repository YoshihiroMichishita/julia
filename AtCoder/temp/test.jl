function YP1()
    N, L = parse.(Int, split(readline()))
    K = parse(Int, readline())
    A = parse.(Int, split(readline()))
    left = 0
    m = L
    id=0
    for it in 1:K
        av = L/(K-it+2)
        #println("av:$(av)")
        id = round(Int, findmin(x -> abs(x-av-left), A)[2])
        a = A[id]
        #println("a:$(a)")
        m2 = a - left
        #println("m2:$(m2)")
        m = min(m,m2)
        left = a
        L -= m2
        #println("m:$(m)")
    end
    m = min(m,L)
    println(m)
end

YP1()