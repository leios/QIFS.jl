/*------------paper_circle----------------------------------------------------//
 Purpose: This creates Figure 2 in the paper
   Notes: After installation, compile with:
              `gcc -o paper_circle paper_circle.c -lquibble`
//----------------------------------------------------------------------------*/

#include <stdio.h>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include "quibble.h"

void create_filename(char *buffer, int i){
    sprintf(buffer, "check%d%d%d%d.csv",
            (i%10000 - i%1000)/1000,
            (i%1000 - i%100)/100,
            (i%100 - i%10)/10,
            i%10);
}

void save_output_array(char *filename, int *arr, int width, int height){

    FILE* file = fopen(filename, "w");

    for (int i = 0; i < width; ++i){
        for (int j = 0; j < height; ++j){
            fprintf(file, "%d", arr[i*width + j]);
            if (j != width - 1){
                fprintf(file, ",");
            }
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

int main(void){

    quibble_program qp = qb_parse_program_file("paper_atom_density.qbl");
    qb_output_program_to_file(qp, "program.txt");
    qb_configure_program(&qp, 1, 0);

    int width = 1024;
    int height = 1024;
    //int width = 128;
    //int height = 128;
    int *arr = (int*)malloc(sizeof(int)*width*height);
    cl_mem d_arr = clCreateBuffer(qp.context,
                                  CL_MEM_READ_WRITE,
                                  width*height * sizeof(int),
                                  NULL,
                                  NULL);

    clEnqueueWriteBuffer(qp.command_queue,
                         d_arr,
                         CL_TRUE,
                         0,
                         width * height * sizeof(int),
                         arr,
                         0,
                         NULL,
                         NULL);


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
                "cl_mem arr", &d_arr,
                "quibble_simple_camera qcam", &qcam,
                 "float time", &time);

    qb_set_args(&qp, "clear_bg_arr", 2,
                "cl_mem arr", &d_arr,
                "quibble_simple_camera qcam", &qcam);

    for (int i = 0; i < num_frames; ++i){

        qb_run(qp, "clear_bg_arr", width*height, 256);
        qb_run(qp, "atom_shader", width*height, 256);

        clEnqueueReadBuffer(qp.command_queue,
                            d_arr,
                            CL_TRUE,
                            0,
                            width*height * sizeof(int),
                            arr,
                            0,
                            NULL,
                            NULL);

        create_filename(filename, i);

        save_output_array(filename, arr, width, height);

        printf("%s\n", filename);

        time += 1/fps;
        qb_set_args(&qp, "atom_shader", 1, "float time", &time);
    }

    qb_free_program(qp);
    clReleaseMemObject(d_arr);
    free(arr);

    return 0;
}
