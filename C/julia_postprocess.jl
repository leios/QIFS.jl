using DelimitedFiles
using Plots
using Images

function make_plots(num_images; maximum = 5000)
    for i = 1:num_images
        input_filename = "check"*lpad(i-1,4,"0")*".csv"
        output_filename = "check_density"*lpad(i-1,4,"0")*".png"
        input_array = readdlm(input_filename, ',', Int, '\n')
        normed_array = input_array / maximum

        output_image = heatmap(normed_array; aspect_ratio=1, colorbar=false, axis=false, size=(1024, 1024), clims=(0,1))

        save(output_filename, output_image)

    end
end
