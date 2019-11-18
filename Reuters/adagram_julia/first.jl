println("Hello world")
using AdaGram

vm, dict = load_model("reuters_model_multisense")
f = open("reuters_polysemy_text2.txt")
lines = readlines(f)
a =[2,3,4]
for i in 1:3
    println(a[i])
end
println(a[1])
for i in 0:1:3
    println(i)
end

println(length(lines))
println(typeof(dict))
println(typeof(vm))
open("Reuters_polysemous_words_array.txt", "a") do f
    for i in 1:1:length(lines)
        println(i)
        println(lines[i])
        each_line = lines[i]
        split_array = split(lines[i])
        for each_word in split_array
            write(f,each_word)
            write(f,"\n")
            writedlm(f, disambiguate(vm, dict, each_word, split_array))
        end
    end
end