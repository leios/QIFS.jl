#=-----------------------------------------------------------------------------#
    combine into video with:
        ffmpeg -r 30 -i check%04d.png -c:v libx264 -pix_fmt yuv420p output.mp4
#-----------------------------------------------------------------------------=#
using Images

function nucleus(circle_radius = 0.3)
end

function electrons(time; beta = 1, circle_radius = 0.3)
end

function make_animation(; final_time = 1)
    output_image = zeros(1000,1000)

    fps = 30
    time = 0

    i = 0
    while time < final_time
        filename = "check"*lpad(i, 4, "0")*".png"
        println(filename)
        save(filename, output_image)
        i += 1
        time += 1/fps
    end 
end
