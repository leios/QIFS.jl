/*------------paper_circle----------------------------------------------------//
 Purpose: This creates Figure 2 in the paper
   Notes: After installation, compile with:
              `gcc -o paper_circle paper_circle.c -lquibble`
//----------------------------------------------------------------------------*/

#include <stdio.h>

#include "quibble.h"

void create_filename(char *buffer, int i){
    sprintf(buffer, "check%d%d%d%d.png",
            (i%10000 - i%1000)/1000,
            (i%1000 - i%100)/100,
            (i%100 - i%10)/10,
            i%10);
}

int main(void){

    quibble_program qp = qb_parse_program_file("paper_atom.qbl");
    qb_output_program_to_file(qp, "program.txt");
    qb_configure_program(&qp, 1, 0);

    int width = 1024;
    int height = 1024;
    //int width = 128;
    //int height = 128;
    quibble_pixels qpix = 
        qb_create_pixel_array(qp, width, height, PRGBA8888);

    float fps = 30;

    // 1 second of frames
    int num_frames = fps*1;

    float world_size_x = 6;
    float ppu = width / world_size_x;
    float world_size_y = height/ppu;

    float world_position_x = 0;
    float world_position_y = 0;

    quibble_simple_camera qcam = qb_create_simple_camera(ppu,
                                                         world_size_x,
                                                         world_size_y,
                                                         world_position_x,
                                                         world_position_y);

    quibble_point_2D location = qb_point_2D(0,0);
    quibble_point_2D velocity = qb_point_2D(0,0);

    float time = 0;
    char filename[15] = {0};
    
    qb_set_args(&qp, "atom_shader", 3,
                "quibble_pixels_prgba8888 qps", qpix,
                "quibble_simple_camera qcam", &qcam,
                 "float time", &time);

    qb_set_args(&qp, "clear_bg", 2,
                "quibble_pixels_prgba8888 qps", qpix,
                "quibble_simple_camera qcam", &qcam);

    for (int i = 0; i < num_frames; ++i){

        qb_run(qp, "clear_bg", width*height, 256);
        qb_run(qp, "atom_shader", width*height, 256);

        qb_pixels_device_to_host(qpix);

        create_filename(filename, i);

        qb_write_png_file(filename, qpix);

        printf("%s\n", filename);

        time += 1/fps;
        qb_set_args(&qp, "atom_shader", 1, "float time", &time);
    }

    qb_free_program(qp);
    qb_free_pixels(qpix);

    return 0;
}
