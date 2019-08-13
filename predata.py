import os


# def y4m2bmp(file_path, save_path):
#     for filename in os.listdir(file_path):
#         subpath = filename.split('.')[0]
#         path = os.path.join(save_path, subpath)
#         if not os.path.exists(path):
#             os.makedirs(path)
#         finishcode = "ffmpeg -i " + os.path.join(file_path, filename) + " -vsync 0 " + \
#                      os.path.join(path, "%3d.bmp") + " -y"
#         print(finishcode)
#         os.system(finishcode)

#
def y4m2bmp(file_path, save_path):
    for filename in os.listdir(file_path):
        finishcode = "ffmpeg -i " + os.path.join(file_path, filename) + " -vsync 0 " + \
                     os.path.join(save_path, filename.replace(".y4m", "%3d.bmp")) + " -y"
        print(finishcode)
        os.system(finishcode)

def bmp2y4m(file_path, save_path):
    for filename in os.listdir(file_path):
        subpath = os.path.join(file_path, filename)
        for i in range(200, 249):
            finishcode = "ffmpeg -i " + subpath + "\%3d.bmp  -pix_fmt yuv420p  -vsync 0 " + \
                         save_path + "\Youku_00" + str(i) + "_h_Res.y4m -y"
            print(finishcode)
            os.system(finishcode)


def select25y4m(file_path, save_path):
    for filename in os.listdir(file_path):
        finishcode = "ffmpeg -i " + os.path.join(file_path, filename) + \
                     " -vf select='not(mod(n\,25))' -vsync 0  -y " + \
                     os.path.join(save_path, filename.replace("_h_Res", "_h_Sub25_Res"))
        print(finishcode)
        os.system(finishcode)


if __name__ == "__main__":
    train_h_res = ".\dataset\\train\\h_res"
    train_h_bmp = ".\dataset\\train\\h_res_bmp"
    train_l_res = ".\dataset\\train\\l_res"
    train_l_bmp = ".\dataset\\train\\l_res_bmp"

    val_h_res = ".\dataset\\val\\h_res"
    val_h_bmp = ".\dataset\\val\\h_res_bmp"
    val_l_res = ".\dataset\\val\\l_res"
    val_l_bmp = ".\dataset\\val\\l_res_bmp"

    test_l_res = ".\dataset\\test\\l_res"
    test_l_bmp = ".\dataset\\test\\l_res_bmp"

    # y4m2bmp(train_h_res, train_h_bmp)
    # y4m2bmp(train_l_res, train_l_bmp)
    # y4m2bmp(val_h_res, val_h_bmp)
    # y4m2bmp(val_l_res, val_l_bmp)
    # y4m2bmp(test_l_res, test_l_bmp)

    output = ".\dataset\\test\\res_y4m"
    # bmp2y4m(test_l_bmp, output)

    sub25output = ".\dataset\\test\\res_sub25_y4m"
    # select25y4m(output, sub25output)
    # y4m2bmp(test_l_res, test_l_bmp)
    y4m2bmp("./y4m6", "./y4m6")

