/* Here are three different test images */

#define IMG_DATA {0, 0, 255, 73, 0, 24, 0, 30, 0, 113, 0, 169, 0, 0, 199, 0, 0, 13, 85, 0, 0, 10, 70, 0, 147, 0, 0, 0, 171, 255, 140, 203, 125, 89, 0, 0, 86, 0, 198, 0, 0, 23, 170, 113, 239, 0, 0, 0, 170, 128, 0, 0, 255, 0, 0, 85, 12, 72, 0, 23, 0, 37, 249, 0, 188, 150, 0, 0, 255, 204, 0, 0, 0, 15, 70, 255, 206, 0, 37, 50, 0, 0, 125, 22, 157, 10, 0, 162, 192, 9, 0, 0, 17, 0, 93, 0, 168, 0, 189, 173, 0, 0, 90, 0, 0, 69, 144, 255, 7, 0, 0, 24, 120, 255, 107, 8, 0, 0, 129, 0, 68, 0, 9, 44, 0, 193, 64, 233, 0, 255, 0, 0, 0, 76, 61, 255, 0, 37, 0, 14, 0, 143, 179, 0, 0, 0, 0, 0, 62, 80, 178, 6, 242, 255, 0, 0, 255, 0, 70, 255, 183, 0, 0, 0, 0, 0, 255, 0, 113, 36, 83, 0, 0, 0, 0, 0, 255, 0, 0, 175, 217, 33, 0, 0, 89, 197, 0, 0, 0, 94, 0, 0, 193, 84, 220, 178, 0, 72, 0, 0, 44, 0, 134, 0, 138, 46, 47, 36, 0, 255, 0, 26, 149, 0, 0, 219, 0, 15, 96, 0, 0, 0, 17, 0, 111, 69, 130, 123, 175, 0, 0, 0, 255, 0, 0, 0, 0, 0, 102, 26, 92, 161, 0, 35, 37, 45, 0, 0, 51, 233, 0, 74, 96, 0, 206, 214, 0, 158, 61, 255, 0, 0, 0, 0, 0, 0, 125, 0, 255, 0, 60, 43, 0, 0, 0, 135, 9, 98, 150, 171, 0, 0, 0, 0, 255, 0, 55, 110, 0, 0, 20, 51, 130, 80, 6, 0, 0, 57, 0, 255, 0, 0, 0, 105, 160, 0, 112, 0, 0, 0, 0, 45, 90, 66, 0, 64, 0, 0, 47, 0, 190, 164, 0, 0, 255, 149, 35, 0, 0, 13, 0, 187, 152, 154, 131, 28, 0, 0, 0, 0, 4, 0, 0, 198, 143, 0, 56, 0, 204, 125, 0, 0, 146, 116, 0, 0, 0, 0, 16, 0, 55, 0, 169, 143, 0, 0, 76, 3, 96, 13, 194, 0, 14, 0, 69, 0, 0, 0, 227, 107, 0, 180, 232, 114, 0, 8, 0, 0, 238, 0, 0, 200, 0, 226, 0, 15, 0, 151, 255, 0, 0, 0, 0, 0, 89, 0, 151, 2, 0, 121, 0, 0, 255, 0, 207, 165, 0, 0, 0, 0, 78, 0, 204, 0, 158, 179, 51, 255, 0, 0, 0, 91, 88, 0, 0, 53, 153, 0, 127, 138, 118, 255, 35, 0, 0, 14, 0, 0, 0, 255, 132, 217, 0, 131, 0, 0, 7, 0, 255, 0, 112, 0, 77, 17, 0, 171, 0, 0, 106, 0, 240, 0, 27, 53, 10, 216, 0, 122, 31, 0, 17, 0, 255, 150, 10, 0, 0, 0, 109, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 58, 23, 51, 24, 0, 171, 0, 175, 139, 0, 83, 255, 0, 95, 255, 255, 51, 10, 0, 11, 0, 7, 138, 112, 49, 0, 74, 48, 0, 0, 0, 171, 180, 7, 73, 171, 57, 0, 0, 0, 244, 12, 55, 95, 0, 0, 127, 0, 84, 174, 0, 0, 64, 104, 127, 47, 0, 0, 0, 0, 172, 38, 0, 0, 68, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 151, 123, 0, 0, 255, 126, 22, 85, 0, 0, 0, 248, 147, 86, 213, 55, 0, 120, 98, 0, 187, 0, 0, 179, 0, 0, 23, 0, 0, 0, 0, 0, 63, 0, 114, 73, 0, 162, 0, 0, 0, 0, 147, 154, 50, 0, 0, 0, 160, 87, 73, 0, 18, 0, 40, 0, 0, 0, 234, 93, 0, 204, 133, 0, 0, 140, 0, 0, 94, 196, 0, 30, 0, 89, 82, 0, 73, 129, 202, 50, 0, 0, 0, 35, 112, 103, 100, 0, 0, 255, 0, 0, 127, 0, 189, 199, 0, 0, 0, 0, 0, 0, 125, 0, 25, 70, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 160, 0, 0, 180, 54, 0, 0, 28, 0, 0, 0, 0, 147, 66, 0, 0, 0, 7, 60, 0, 0, 0, 255, 45, 43, 0, 0, 255, 0, 38, 153, 78, 0, 86, 55, 0, 0, 66, 0, 0, 1, 0, 101, 138, 221, 204, 0, 0, 0, 0, 155, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 61, 83, 0, 0, 157, 0, 96, 155, 0, 133, 112, 0, 155, 117, 206, 157, 35, 158, 0, 0, 0, 163, 255, 0, 0, 101, 255, 153, 0, 0, 78, 92, 0, 249, 41, 212, 0, 0, 0, 187, 13, 109, 71, 0, 0, 180, 15, 211, 238, 237, 50, 33, 190, 86, 240, 0, 47, 41, 51, 173, 110, 76, 86, 178, 0, 0, 67, 0, 255, 0, 255, 0, 7, 86, 0, 197, 136, 0, 164, 18, 156, 150, 0, 0, 245, 189, 175, 67, 44, 162, 29, 0, 0, 169, 120, 95, 103, 0, 229, 43, 0, 0, 162, 0, 0, 0, 0, 101, 49, 41, 160, 0, 0, 0, 0, 0, 35, 10, 0, 65, 0, 0, 125, 97, 121, 161, 0, 109, 0, 103, 13, 0, 220, 178, 147, 0, 121, 45, 206, 0, 51, 0, 0, 0, 67, 0, 128, 0, 107, 0, 145, 118, 206, 198, 0, 0, 0, 0, 0, 0, 30, 255, 224, 0, 6, 0, 0, 21, 0, 255, 164, 0, 0, 0, 0, 0, 0, 191, 29, 0, 102, 0, 67, 0, 108, 10, 0, 255, 0, 32, 242, 198, 0, 50, 193, 108, 245, 70, 5, 29, 0, 0, 0, 6, 19, 0, 19, 0, 58, 0, 0, 103, 0, 0, 0, 93, 87, 113, 0, 0, 153, 0, 172, 71, 167, 177, 32, 134, 101, 0, 12, 0, 59, 153, 0, 148, 0, 41, 45, 147, 2, 0, 0, 53, 0, 12, 0, 155, 0, 56, 0, 107, 0, 40, 0, 108, 120, 65, 122, 97, 229, 41, 188, 250, 96, 255, 175, 0, 98, 229, 28, 74, 0, 47, 0, 0, 0, 0, 255, 240, 0, 0, 182, 0, 0, 0, 0, 0, 0, 8, 102, 0, 0, 0, 0, 112, 70, 129, 0, 15, 0, 196, 0, 164, 152, 0, 15, 0, 239, 92, 213, 7, 0, 125, 23, 245, 8, 142, 30, 255, 0, 0, 0, 42, 224, 0, 0, 11, 0, 110, 167, 215, 146, 0, 0, 0, 135, 0, 0, 0, 141, 0, 255, 87, 0, 231, 0, 57, 0, 232, 0, 38, 0, 0, 80, 210, 10, 0, 11, 0, 0, 0, 0, 125, 48, 175, 3, 0, 0, 0, 0, 0, 0, 179, 167, 54, 0, 6, 0, 53, 0, 70, 0, 84, 11, 85, 0, 0, 0, 179, 111, 0, 57, 0, 0, 0, 203, 0, 0, 0, 19, 11, 59, 0, 216, 0, 0, 84, 255, 151, 5, 0, 100, 0, 140, 0, 61, 248, 0, 0, 0, 0, 0, 1, 0, 94, 95, 255, 0, 0, 26, 0, 0, 255, 0, 0, 208, 0, 56, 118, 0, 0, 0, 255, 0, 0, 0, 160, 0, 33, 134, 255, 255, 156, 78, 109, 0, 0, 0, 0, 0, 45, 60, 0, 43, 0, 0, 67, 0, 72, 0, 152, 101, 0, 117, 0, 126, 0, 3, 134, 154, 0, 117, 0, 112, 0, 21, 0, 15, 255, 0, 133, 32, 0, 150, 104, 0, 0, 0, 0, 0, 21, 0, 92, 8, 0, 63, 33, 10, 0, 0, 255, 0, 115, 6, 106, 37, 0, 70, 194, 95, 225, 255, 0, 170, 197, 43, 0, 26, 0, 0, 0, 176, 5, 38, 255, 0, 0, 0, 0, 0, 0, 0, 22, 224, 40, 21, 0, 213, 0, 120, 158, 0, 85, 12, 95, 0, 184, 0, 0, 37, 255, 0, 0, 198, 113, 165, 59, 0, 0, 0, 166, 118, 0, 0, 255, 0, 0, 0, 178, 255, 26, 89, 0, 0, 189, 0, 182, 147, 0, 0, 255, 191, 0, 0, 0, 0, 9, 220, 191, 0, 174, 109, 81, 87, 246, 0, 14, 0, 0, 197, 164, 0, 0, 0, 82, 0, 0, 35, 134, 43, 255, 0, 0, 0, 0, 11, 0, 28, 135, 24, 34, 0, 79, 0, 9, 235, 62, 48, 0, 0, 51, 0, 34, 0, 33, 207, 0, 208, 0, 0, 60, 197, 0, 0, 76, 230, 0, 61, 0, 249, 0, 0, 0, 35, 140, 20, 0, 0, 136, 0, 0, 99, 164, 0, 0, 180, 0, 0, 255, 0, 110, 255, 0, 0, 0, 0, 0, 0, 65, 0, 153, 244, 20, 196, 0, 0, 0, 0, 32, 0, 16, 25, 90, 0, 42, 0, 55, 220, 223, 0, 4, 107, 0, 0, 0, 0, 2, 95, 0, 0, 0, 0, 3, 0, 0, 7, 213, 67, 50, 117, 62, 189, 0, 0, 201, 111, 14, 255, 0, 0, 0, 255, 0, 77, 80, 0, 13, 0, 85, 0, 0, 0, 0, 0, 232, 0, 106, 0, 4, 0, 0, 22, 0, 0, 0, 0, 113, 23, 71, 0, 0, 172, 0, 0, 131, 27, 231, 255, 0, 255, 177, 90, 10, 0, 0, 0, 13, 0, 0, 192, 255, 0, 56, 128, 0, 1, 149, 183, 204, 137, 39, 199, 39, 0, 0, 0, 250, 0, 30, 33, 20, 0, 0, 181, 0, 28, 168, 0, 0, 110, 0, 66, 189, 0, 115, 115, 108, 129, 190, 0, 206, 0, 0, 0, 0, 241, 0, 0, 0, 0, 196, 34, 30, 131, 0, 0, 161, 182, 57, 0, 56, 50, 25, 255, 255, 51, 198, 63, 0, 49, 169, 0, 79, 45, 0, 196, 0, 0, 139, 0, 139, 0, 0, 0, 44, 73, 255, 0, 0, 137, 0, 0, 0, 0, 255, 255, 150, 0, 14, 0, 92, 255, 250, 0, 62, 107, 207, 0, 65, 0, 209, 0, 99, 166, 101, 0, 0, 255, 0, 0, 18, 42, 0, 255, 0, 0, 0, 0, 18, 153, 150, 55, 0, 0, 204, 24, 176, 77, 255, 54, 0, 0, 0, 0, 150, 0, 143, 255, 41, 111, 0, 0, 0, 0, 255, 0, 0, 244, 112, 185, 0, 0, 0, 0, 217, 0, 0, 122, 3, 17, 128, 104, 186, 255, 8, 0, 10, 201, 41, 0, 0, 22, 112, 0, 0, 191, 0, 0, 141, 0, 191, 0, 198, 202, 106, 141, 0, 235, 71, 0, 250, 231, 166, 255, 0, 58, 0, 50, 55, 18, 149, 0, 0, 0, 74, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 57, 0, 0, 0, 0, 36, 25, 116, 0, 72, 138, 149, 129, 94, 0, 191, 0, 0, 182, 92, 253, 202, 28, 0, 0, 226, 0, 45, 21, 135, 0, 8, 217, 92, 0, 19, 196, 26, 161, 255, 91, 0, 2, 55, 0, 255, 52, 113, 0, 60, 0, 203, 214, 130, 35, 255, 0, 62, 39, 0, 255, 98, 0, 0, 0, 113, 0, 0, 176, 210, 0, 0, 95, 99, 37, 0, 255, 22, 149, 255, 0, 252, 125, 0, 0, 0, 89, 0, 0, 26, 26, 0, 237, 255, 214, 172, 0, 0, 0, 191, 32, 0, 0, 0, 209, 139, 0, 170, 65, 0, 51, 0, 0, 105, 2, 178, 206, 0, 0, 0, 0, 0, 0, 150, 23, 31, 95, 0, 23, 21, 255, 207, 0, 46, 0, 78, 0, 0, 3, 79, 191, 126, 255, 173, 160, 0, 255, 75, 38, 19, 38, 0, 65, 0, 255, 99, 0, 121, 127, 220, 133, 0, 89, 78, 0, 0, 145, 145, 0, 27, 122, 0, 0, 186, 0, 234, 192, 0, 0, 0, 171, 0, 0, 32, 0, 0, 255, 77, 195, 0, 0, 0, 84, 55, 75, 0, 105, 12, 0, 55, 0, 47, 187, 139, 0, 8, 0, 0, 65, 131, 30, 0, 81, 0, 175, 0, 0, 39, 0, 0, 141, 45, 91, 49, 50, 0, 210, 65, 0, 15, 0, 198, 205, 0, 0, 0, 255, 197, 71, 58, 0, 46, 82, 255, 221, 149, 0, 0, 0, 120, 26, 6, 0, 83, 10, 0, 45, 0, 0, 0, 0, 230, 201, 135, 48, 146, 72, 59, 172, 96, 106, 225, 0, 0, 178, 242, 247, 172, 39, 194, 0, 0, 0, 169, 204, 255, 0, 26, 0, 0, 0, 40, 41, 147, 52, 19, 2, 58, 0, 0, 100, 71, 0, 154, 107, 0, 0, 137, 0, 172, 113, 116, 0, 0, 110, 69, 151, 0, 0, 0, 0, 174, 48, 15, 14, 20, 0, 0, 0, 69, 120, 0, 155, 145, 0, 0, 0, 113, 150, 0, 0, 255, 38, 0, 0, 68, 209, 0, 182, 106, 0, 67, 112, 0, 0, 215, 0, 90, 0, 0, 63, 184, 0, 0, 0, 78, 39, 0, 0, 161, 32, 0, 179, 0, 255, 75, 0, 0, 0, 130, 168, 118, 0, 8, 0, 234, 177, 108, 0, 0, 0, 144, 0, 0, 0, 121, 195, 0, 230, 174, 204, 0, 218, 0, 0, 0, 0, 29, 182, 0, 255, 147, 17, 0, 124, 223, 44, 0, 0, 63, 0, 25, 255, 28, 0, 44, 122, 0, 0, 143, 0, 107, 255, 27, 0, 0, 70, 0, 0, 184, 0, 12, 255, 0, 255, 0, 16, 0, 42, 255, 0, 21, 31, 78, 0, 0, 64, 40, 63, 205, 0, 0, 107, 0, 38, 92, 90, 48, 109, 0, 49, 0, 0, 5, 0, 40, 0, 147, 0, 98, 22, 0, 172, 0, 0, 99, 152, 236, 139, 0, 0, 81, 174, 222, 62, 8, 0, 0, 0, 255, 57, 0, 0, 0, 0, 255, 0, 15, 0, 0, 0, 10, 30, 0, 0, 0, 0, 114, 72, 121, 0, 0, 255, 117, 92, 19, 0, 161, 255, 0, 142, 50, 220, 56, 2, 0, 168, 0, 0, 64, 94, 138, 0, 0, 130, 0, 0, 126, 159, 21, 0, 145, 158, 31, 255, 48, 0, 231, 36, 23, 0, 21, 41, 222, 141, 5, 0, 0, 0, 74, 122, 68, 65, 0, 0, 128, 173, 199, 76, 0, 58, 206, 0, 0, 12, 0, 35, 0, 41, 0, 0, 2, 55, 151, 177, 0, 0, 255, 253, 66, 21, 69, 34, 0, 99, 242, 230, 255, 6, 107, 87, 222, 30, 247, 0, 0, 60, 18, 0, 124, 0, 0, 222, 0, 0, 93, 115, 255, 139, 0, 0, 56, 0, 0, 130, 255, 0, 0, 71, 0, 114, 147, 0, 170, 16, 53, 70, 39, 0, 127, 0, 225, 204, 0, 179, 11, 127, 0, 164, 0, 0, 0, 231, 0, 243, 0, 255, 168, 139, 40, 101, 96, 0, 55, 49, 180, 0, 5, 86, 51, 0, 122, 255, 0, 0, 173, 0, 158, 255, 0, 0, 184, 0, 18, 0, 223, 0, 0, 255, 130, 255, 140, 0, 0, 0, 225, 0, 0, 255, 0, 0, 89, 246, 163, 90, 184, 12, 0, 11, 32, 0, 0, 172, 198, 218, 0, 0, 0, 0, 3, 31, 91, 0, 126, 0, 52, 9, 0, 179, 0, 58, 255, 76, 41, 155, 195, 140, 0, 25, 0, 0, 127, 60, 0, 0, 11, 49, 216, 0, 0, 0, 0, 0, 65, 0, 0, 0, 2, 0, 10, 0, 0, 255, 40, 150, 175, 63, 0, 174, 0, 104, 8, 0, 72, 229, 0, 97, 255, 92, 0, 136, 22, 0, 0, 0, 99, 255, 0, 0, 3, 219, 8, 0, 0, 138, 196, 116, 0, 89, 255, 0, 85, 0, 224, 74, 146, 161, 0, 78, 49, 0, 0, 184, 138, 0, 0, 155, 255, 176, 55, 146, 143, 69, 255, 163, 11, 0, 84, 0, 0, 0, 65, 166, 0, 255, 0, 0, 27, 70, 49, 162, 0, 70, 60, 113, 0, 0, 0, 0, 210, 255, 194, 57, 255, 0, 112, 0, 255, 0, 3, 0, 0, 86, 255, 0, 149, 0, 0, 6, 0, 0, 115, 135, 135, 47, 0, 208, 0, 130, 0, 0, 168, 0, 0, 0, 0, 0, 255, 255, 31, 0, 48, 207, 229, 0, 0, 0, 158, 154, 0, 140, 5, 0, 0, 113, 0, 0, 0, 255, 102, 255, 0, 0, 75, 0, 0, 198, 227, 0, 0, 0, 110, 58, 0, 155, 249, 12, 0, 255, 0, 83, 103, 0, 48, 255, 26, 43, 0, 0, 23, 0, 245, 36, 0, 255, 114, 219, 0, 0, 0, 0, 164, 0, 28, 22, 255, 0, 91, 0, 0, 0, 0, 0, 8, 255, 0, 0, 49, 180, 157, 156, 0, 0, 0, 0, 0, 0, 0, 0, 100, 0, 146, 170, 0, 186, 0, 48, 229, 101, 70, 251, 0, 0, 0, 32, 0, 150, 77, 57, 0, 0, 0, 0, 0, 99, 0, 0, 121, 0, 30, 0, 0, 0, 94, 0, 0, 0, 53, 0, 22, 20, 184, 0, 0, 213, 157, 110, 255, 52, 255, 220, 0, 0, 255, 199, 74, 0, 0, 0, 0, 14, 56, 194, 82, 0, 31, 0, 0, 0, 180, 166, 91, 0, 0, 108, 156, 32, 56, 0, 255, 152, 0, 0, 0, 31, 60, 0, 0, 170, 168, 0, 0, 85, 141, 39, 0, 0, 0, 150, 212, 3, 224, 0, 41, 0, 0, 175, 5, 76, 0, 47, 0, 0, 206, 0, 10, 156, 0, 17, 0, 0, 0, 0, 0, 22, 26, 255, 168, 28, 255, 180, 0, 0, 255, 0, 37, 0, 0, 0, 157, 0, 216, 0, 236, 60, 17, 51, 137, 214, 255, 3, 0, 124, 0, 16, 0, 45, 191, 10, 0, 0, 116, 87, 215, 214, 245, 0, 51, 0, 121, 0, 161, 36, 64, 94, 255, 187, 161, 0, 165, 10, 0, 0, 0, 6, 11, 125, 0, 0, 132, 45, 0, 121, 255, 0, 0, 58, 81, 0, 0, 145, 255, 1, 0, 246, 210, 194, 116, 0, 121, 255, 5, 27, 0, 0, 0, 0, 0, 0, 0, 111, 167, 0, 0, 0, 0, 0, 44, 0, 0, 0, 140, 0, 130, 66, 50, 0, 255, 8, 152, 72, 0, 0, 0, 0, 237, 107, 0, 97, 0, 0, 88, 0, 147, 0, 116, 88, 0, 140, 0, 95, 0, 0, 255, 147, 173, 255, 255, 0, 79, 140, 0, 107, 66, 0, 0, 124, 71, 41, 153, 0, 0, 0, 221, 0, 0, 0, 0, 0, 115, 0, 40, 0, 0, 0, 115, 122, 255, 0, 0, 156, 100, 255, 254, 0, 217, 255, 1, 253, 255, 182, 158, 0, 0, 0, 0, 0, 113, 255, 0, 0, 76, 156, 0, 0, 0, 11, 124, 89, 0, 0, 255, 0, 0, 0, 113, 195, 163, 161, 61, 134, 174, 0, 168, 66, 206, 0, 0, 158, 107, 170, 193, 0, 0, 59, 234, 66, 0, 24, 213, 0, 0, 0, 0, 255, 0, 90, 0, 0, 73, 0, 210, 154, 0, 47, 135, 241, 111, 70, 0, 126, 0, 255, 255, 20, 135, 33, 0, 119, 208, 29, 1, 29, 0, 87, 0, 57, 79, 0, 135, 0, 0, 0, 118, 0, 0, 100, 0, 0, 0, 31, 0, 0, 255, 128, 0, 0, 44, 147, 154, 255, 75, 0, 64, 0, 52, 0, 0, 0, 215, 209, 0, 205, 193, 206, 0, 110, 0, 0, 0, 227, 0, 79, 0, 0, 87, 125, 0, 147, 114, 90, 58, 0, 102, 126, 132, 202, 255, 0, 0, 0, 0, 69, 70, 0, 208, 255, 2, 0, 0, 0, 0, 0, 255, 0, 0, 255, 5, 1, 0, 0, 0, 0, 235, 0, 0, 0, 187, 124, 28, 94, 45, 255, 213, 0, 11, 205, 51, 0, 25, 193, 191, 62, 0, 28, 0, 200, 46, 0, 0, 0, 255, 84, 0, 0, 0, 255, 0, 131, 130, 255, 163, 255, 111, 48, 0, 0, 48, 0, 126, 0, 239, 0, 186, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 70, 100, 38, 0, 95, 156, 155, 255, 56, 175, 139, 0, 83, 255, 95, 93, 0, 0, 0, 0, 0, 0, 154, 211, 0, 0, 255, 0, 2, 7, 0, 190, 195, 0, 0, 255, 255, 0, 0, 172, 0, 0, 0, 130, 0, 0, 79, 250, 93, 197, 0, 41, 111, 255, 172, 0, 0, 0, 0, 255, 98, 0, 0, 210, 0, 0, 170, 46, 227, 0, 179, 0, 0, 79, 42, 143, 139, 24, 33, 74, 255, 0, 0, 130, 134, 0, 255, 217, 36, 171, 223, 0, 255, 64, 0, 86, 34, 0, 211, 0, 15, 15, 0, 59, 0, 0, 0, 209, 0, 81, 45, 0, 0, 0, 0, 0, 0, 212, 47, 24, 0, 0, 0, 38, 255, 113, 0, 0, 0, 64, 0, 0, 0, 58, 234, 0, 176, 113, 0, 90, 255, 0, 0, 0, 0, 0, 75, 0, 103, 140, 0, 0, 244, 229, 0, 49, 0, 156, 131, 0, 195, 138, 0, 0, 94, 0, 30, 101, 0, 102, 255, 7, 0, 0, 0, 0, 0, 196, 0, 32, 255, 60, 171, 0, 0, 0, 0, 32, 0, 0, 0, 255, 0, 39, 0, 13, 108, 130, 0, 16, 106, 0, 0, 68, 83, 142, 0, 0, 153, 0, 52, 129, 63, 0, 0, 237, 0, 47, 0, 37, 198, 0, 255, 0, 67, 149, 83, 0, 0, 0, 23, 0, 100, 254, 0, 27, 0, 201, 0, 33, 0, 0, 0, 104, 0, 38, 0, 0, 0, 0, 0, 0, 23, 0, 0, 108, 39, 20, 175, 85, 102, 87, 96, 247, 0, 96, 140, 0, 201, 33, 204, 0, 0, 9, 87, 0, 55, 165, 120, 158, 69, 57, 8, 0, 114, 25, 0, 98, 0, 161, 96, 0, 48, 0, 0, 196, 0, 13, 60, 0, 0, 69, 0, 0, 90, 151, 28, 77, 174, 178, 48, 148, 169, 205, 210, 161, 0, 80, 165, 6, 0, 0, 34, 0, 246, 0, 0, 0, 0, 99, 81, 185, 169, 184, 15, 32, 76, 45, 0, 253, 114, 0, 214, 122, 90, 79, 142, 0, 19, 255, 0, 62, 0, 0, 67, 66, 0, 22, 0, 129, 16, 0, 0, 119, 255, 172, 180, 0, 0, 0, 0, 0, 188, 149, 255, 3, 138, 0, 68, 255, 185, 255, 46, 170, 155, 7, 0, 43, 0, 255, 76, 0, 109, 0, 95, 0, 81, 0, 0, 0, 0, 0, 171, 12, 65, 49, 177, 49, 211, 255, 0, 0, 0, 17, 58, 149, 52, 255, 207, 0, 255, 0, 14, 191, 0, 192, 255, 55, 0, 67, 0, 0, 0, 0, 0, 53, 0, 149, 210, 0, 68, 0, 0, 130, 0, 0, 255, 83, 0, 160, 78, 227, 255, 0, 255, 0, 0, 13, 0, 114, 149, 101, 85, 0, 10, 0, 133, 67, 0, 156, 0, 32, 90, 72, 189, 0, 208, 0, 252, 179, 255, 86, 160, 38, 63, 0, 108, 0, 79, 94, 56, 7, 54, 0, 129, 0, 0, 0, 0, 0, 0, 111, 0, 154, 48, 63, 0, 202, 0, 0, 0, 16, 157, 66, 31, 42, 168, 247, 163, 50, 156, 216, 255, 0, 16, 138, 87, 65, 120, 11, 85, 0, 0, 0, 58, 47, 0, 82, 249, 9, 0, 0, 127, 0, 0, 0, 0, 169, 0, 0, 0, 108, 28, 60, 0, 81, 38, 255, 0, 202, 133, 45, 0, 32, 124, 35, 82, 255, 0, 0, 110, 165, 0, 37, 174, 144, 0, 0, 0, 0, 86, 0, 255, 0, 149, 72, 0, 203, 176, 0, 23, 43, 243, 0, 219, 43, 141, 178, 255, 255, 0, 72, 41, 0, 105, 55, 0, 0, 0, 0, 106, 89, 0, 98, 190, 210, 43, 26, 0, 174, 62, 200, 250, 0, 31, 0, 0, 0, 0, 55, 0, 0, 0, 0, 132, 255, 210, 61, 0, 0, 0, 184, 0, 35, 0, 164, 60, 225, 241, 255, 255, 0, 220, 0, 0, 0, 52, 107, 175, 0, 0, 2, 132, 255, 178, 242, 0, 0, 19, 0, 0, 0, 255, 253, 0, 0, 175, 0, 0, 141, 30, 155, 255, 222, 54, 0, 59, 0, 0, 67, 30, 0, 125, 0, 97, 0, 135, 0, 10, 109, 0, 0, 120, 0, 0, 0, 191, 95, 175, 142, 0, 64, 30, 50, 0, 255, 255, 50, 210, 0, 123, 0, 34, 0, 0, 0, 23, 255, 31, 71, 0, 0, 255, 0, 194, 154, 169, 14, 255, 0, 45, 0, 60, 0, 35, 75, 191, 23, 53, 107, 173, 91, 42, 0, 0, 162, 0, 0, 0, 84, 0, 0, 0, 22, 0, 0, 0, 87, 210, 116, 0, 14, 199, 108, 124, 0, 90, 231, 207, 0, 0, 250, 77, 55, 0, 78, 0, 0, 0, 0, 255, 255, 0, 0, 111, 0, 0, 0, 33, 0, 81, 35, 62, 26, 0, 0, 0, 189, 91, 93, 0, 42, 0, 0, 210, 48, 117, 0, 0, 0, 180, 0, 255, 0, 0, 0, 61, 255, 54, 113, 0, 229, 0, 0, 54, 85, 176, 0, 77, 21, 0, 0, 0, 64, 133, 0, 0, 255, 163, 11, 0, 0, 121, 0, 255, 73, 0, 225, 77, 0, 62, 235, 0, 88, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 117, 0, 82, 83, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 90, 77, 0, 0, 0, 78, 0, 0, 0, 5, 106, 0, 121, 44, 161, 0, 48, 0, 0, 248, 93, 0, 255, 0, 255, 0, 0, 0, 218, 182, 75, 0, 138, 57, 0, 0, 242, 122, 0, 0, 0, 0, 0, 114, 0, 0, 212, 0, 0, 0, 0, 0, 0, 255, 0, 50, 252, 0, 255, 10, 0, 0, 0, 210, 0, 2, 147, 104, 0, 0, 228, 130, 255, 4, 0, 0, 97, 0, 0, 0, 0, 113, 38, 0, 137, 0, 0, 142, 0, 0, 0, 176, 0, 0, 1, 0, 226, 0, 0, 170, 0, 187, 4, 0, 0, 39, 0, 0, 55, 203, 0, 241, 0, 223, 0, 71, 0, 0, 0, 255, 0, 20, 0, 174, 0, 0, 11, 0, 55, 0, 4, 0, 0, 113, 0, 0, 184, 0, 111, 254, 0, 30, 65, 0, 0, 255, 91, 71, 37, 255, 0, 43, 0, 0, 201, 255, 0, 141, 0, 0, 0, 245, 7, 162, 2, 0, 63, 120, 0, 160, 12, 149, 195, 168, 196, 45, 0, 125, 0, 232, 131, 141, 0, 2, 136, 74, 117, 205, 34, 0, 13, 221, 0, 123, 142, 43, 0, 0, 0, 36, 158, 0, 0, 0, 0, 124, 114, 100, 156, 19, 0, 182, 85, 41, 0, 0, 0, 151, 122, 229, 80, 106, 118, 252, 0, 119, 54, 59, 0, 0, 121, 0, 0, 0, 0, 43, 35, 27, 0, 139, 179, 255, 0, 0, 190, 0, 0, 0, 62, 72, 120, 205, 0, 255, 2, 255, 209, 170, 0, 0, 0, 152, 0, 0, 0, 173, 0, 39, 70, 137, 0, 0, 120, 0, 0, 166, 84, 0, 255, 0, 101, 110, 133, 10, 186, 255, 0, 0, 96, 101, 0, 0, 180, 89, 4, 94, 225, 34, 81, 255, 0, 157, 255, 255, 0, 67, 10, 30, 0, 152, 0, 40, 215, 197, 188, 0, 134, 0, 0, 210, 143, 0, 28, 255, 0, 76, 11, 25, 63, 255, 204, 0, 96, 0, 0, 0, 0, 93, 245, 0, 124, 0, 0, 3, 0, 0, 0, 0, 29, 141, 111, 0, 150, 0, 0, 81, 207, 255, 102, 124, 172, 36, 92, 194, 29, 0, 0, 165, 0, 0, 113, 255, 0, 46, 0, 54, 0, 0, 0, 0, 188, 146, 163, 19, 89, 0, 1, 0, 255, 0, 0, 0, 132, 84, 27, 255, 48, 249, 169, 0, 219, 0, 156, 161, 14, 119, 0, 0, 0, 69, 32, 251, 0, 98, 165, 0, 0, 0, 122, 217, 13, 0, 171, 0, 0, 0, 0, 219, 37, 0, 0, 105, 0, 0, 15, 107, 0, 193, 0, 0, 99, 43, 255, 0, 0, 50, 181, 255, 63, 90, 10, 255, 0, 0, 23, 0, 50, 0, 0, 0, 0, 0, 0, 255, 151, 0, 18, 73, 119, 0, 0, 0, 121, 0, 52, 72, 100, 255, 54, 39, 64, 118, 211, 0, 54, 0, 201, 29, 0, 70, 0, 166, 34, 3, 0, 100, 29, 91, 17, 0, 0, 0, 0, 0, 0, 255, 57, 0, 3, 255, 0, 167, 222, 83, 0, 0, 255, 21, 0, 103, 0, 190, 21, 37, 53, 247, 15, 0, 255, 26, 0, 202, 0, 0, 40, 0, 144, 0, 49, 0, 215, 201, 0, 0, 36, 10, 236, 67, 114, 255, 0, 0, 105, 0, 15, 159, 0, 47, 148, 126, 0, 3, 0, 0, 0, 215, 0, 0, 189, 0, 255, 0, 0, 0, 0, 0, 0, 0, 212, 174, 0, 91, 0, 121, 72, 42, 57, 0, 102, 0, 0, 0, 35, 3, 138, 0, 190, 0, 0, 60, 0, 56, 0, 0, 39, 166, 106, 58, 191, 0, 58, 66, 0, 212, 145, 70, 0, 39, 99, 0, 38, 153, 0, 0, 0, 79, 5, 255, 0, 0, 0, 163, 0, 0, 0, 0, 0, 36, 0, 0, 0, 0, 34, 215, 0, 66, 0, 0, 149, 0, 92, 58, 0, 150, 109, 47, 255, 245, 71, 98, 48, 0, 0, 139, 166, 10, 213, 255, 0, 0, 127, 0, 0, 0, 0, 0, 157, 173, 0, 7, 0, 0, 63, 60, 0, 78, 42, 20, 0, 255, 85, 29, 0, 255, 0, 122, 151, 54, 154, 167, 3, 0, 0, 250, 125, 0, 20, 255, 0, 0, 0, 0, 255, 0, 150, 93, 25, 255, 81, 66, 119, 0, 35, 89, 38, 67, 23, 39, 152, 0, 188, 1, 130, 145, 0, 27, 0, 56, 0, 53, 0, 0, 145, 107, 0, 117, 0, 0, 0, 0, 0, 132, 9, 166, 72, 0, 0, 0, 0, 24, 212, 255, 39, 5, 0, 0, 0, 142, 249, 170, 0, 0, 0, 77, 0, 0, 0, 163, 143, 0, 255, 0, 46, 0, 255, 0, 0, 0, 221, 0, 155, 0, 194, 28, 4, 0, 236, 205, 0, 0, 0, 13, 7, 66, 110, 218, 0, 0, 90, 0, 0, 182, 0, 170, 255, 229, 0, 0, 20, 0, 0, 248, 0, 0, 90, 0, 255, 0, 38, 0, 0, 255, 0, 11, 255, 255, 0, 31, 0, 119, 150, 121, 0, 97, 23, 9, 30, 0, 4, 110, 165, 0, 242, 6, 0, 0, 0, 180, 142, 19, 69, 0, 0, 37, 200, 0, 122, 217, 168, 0, 187, 69, 77, 0, 204, 0, 49, 169, 0, 49, 98, 0, 0, 74, 0, 0, 0, 69, 0, 72, 0, 255, 40, 104, 125, 131, 0, 0, 0, 154, 47, 0, 0, 0, 151, 87, 65, 243, 135, 159, 187, 0, 0, 181, 58, 42, 0, 0, 0, 0, 0, 1, 37, 255, 0, 15, 0, 0, 0, 0, 0, 30, 0, 0, 59, 0, 0, 0, 0, 0, 0, 84, 0, 0, 44, 226, 0, 0, 177, 159, 0, 0, 56, 130, 60, 0, 52, 0, 0, 146, 96, 68, 0, 0, 0, 0, 0, 0, 191, 0, 219, 0, 0, 0, 72, 153, 131, 0, 0, 22, 110, 0, 44, 0, 63, 0, 214, 163, 0, 67, 107, 0, 22, 217, 0, 94, 0, 0, 0, 75, 0, 113, 0, 138, 0, 0, 0, 125, 25, 38, 230, 0, 150, 0, 0, 0, 0, 137, 114, 107, 0, 0, 37, 194, 221, 105, 0, 29, 0, 108, 0, 0, 0, 156, 19, 0, 106, 0, 0, 0, 161, 0, 0, 10, 0, 0, 135, 0, 0, 0, 40, 132, 175, 112, 0, 0, 0, 16, 47, 0, 0, 254, 0, 0, 0, 0, 0, 52, 0, 183, 255, 0, 0, 0, 0, 0, 0, 255, 0, 0, 244, 0, 250, 0, 0, 0, 0, 255, 0, 0, 0, 9, 0, 76, 49, 33, 198, 62, 0, 0, 78, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 155, 0, 129, 0, 225, 0, 0, 100, 0, 215, 0, 6, 128, 82, 255, 69, 0, 1, 0, 110, 0, 98, 132, 0, 0, 101, 255, 60, 52, 0, 0, 0, 203, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 30, 0, 9, 198, 103, 124, 48, 17, 110, 251, 0, 220, 255, 49, 98, 8, 0, 0, 19, 87, 0, 255, 216, 0, 26, 0, 75, 0, 255, 0, 199, 0, 93, 125, 0, 0, 0, 0, 146, 237, 53, 0, 0, 15, 141, 0, 133, 0, 90, 0, 3, 49, 114, 36, 0, 19, 0, 0, 32, 30, 128, 0, 171, 0, 22, 0, 0, 253, 0, 18, 0, 0, 13, 4, 132, 117, 0, 0, 0, 88, 0, 166, 0, 5, 0, 155, 215, 221, 90, 88, 0, 97, 255, 0, 119, 0, 0, 67, 0, 0, 0, 0, 0, 0, 0, 0, 111, 33, 35, 55, 0, 152, 0, 0, 0, 223, 189, 49, 40, 0, 86, 86, 230, 218, 2, 0, 140, 0, 188, 0, 21, 0, 229, 93, 166, 110, 255, 0, 21, 152, 0, 245, 187, 0, 0, 0, 0, 0, 132, 84, 57, 183, 166, 0, 0, 76, 161, 36, 0, 110, 153, 0, 0, 13, 0, 0, 0, 129, 77, 211, 94, 0, 0, 0, 0, 0, 0, 0, 0, 255, 112, 255, 137, 15, 0, 0, 158, 0, 0, 255, 75, 0, 25, 70, 0, 55, 0, 0, 62, 47, 0, 0, 3, 39, 12, 17, 0, 0, 0, 0, 125, 0, 7, 0, 227, 159, 0, 122, 0, 254, 0, 30, 118, 107, 219, 3, 193, 0, 0, 0, 0, 0, 206, 57, 30, 0, 0, 0, 190, 0, 0, 0, 255, 0, 0, 0, 0, 0, 117, 0, 0, 0, 0, 0, 88, 60, 64, 0, 0, 149, 4, 8, 147, 0, 255, 141, 72, 103, 50, 139, 189, 1, 255, 0, 64, 0, 0, 217, 255, 0, 107, 177, 0, 0, 0, 118, 10, 111, 0, 150, 80, 84, 28, 0, 84, 69, 0, 0, 66, 0, 23, 45, 138, 119, 255, 0, 0, 158, 159, 182, 131, 0, 0, 58, 221, 0, 218, 195, 253, 0, 66, 121, 26, 255, 0, 64, 0, 10, 93, 0, 255, 155, 0, 135, 52, 255, 0, 0, 0, 255, 0, 152, 173, 202, 209, 175, 0, 0, 120, 206, 0, 25, 0, 230, 106, 238, 0, 84, 132, 0, 0, 0, 128, 180, 77, 0, 0, 0, 31, 0, 0, 0, 85, 0, 0, 2, 0, 0, 99, 208, 255, 0, 20, 0, 90, 0, 0, 255, 164, 81, 0, 87, 0, 158, 0, 87, 0, 0, 0, 30, 0, 0, 0, 212, 125, 62, 0, 209, 113, 0, 91, 0, 123, 118, 0, 240, 234, 10, 37, 164, 0, 197, 235, 42, 10, 255, 77, 86, 0, 0, 0, 2, 165, 0, 0, 255, 23, 255, 100, 0, 0, 0, 108, 176, 0, 0, 227, 0, 94, 0, 148, 205, 255, 56, 64, 0, 0, 109, 201, 131, 182, 255, 0, 239, 0, 6, 0, 0, 0, 0, 255, 111, 37, 0, 47, 70, 0, 0, 110, 255, 83, 255, 0, 0, 0, 39, 0, 168, 52, 0, 60, 0, 0, 0, 233, 0, 0, 9, 0, 0, 0, 0, 0, 0, 89, 41, 25, 0, 0, 0, 173, 187, 149, 0, 0, 182, 0, 255, 194, 175, 164, 255, 35, 5, 167, 226, 6, 0, 0, 0, 0, 0, 57, 157, 128, 41, 15, 116, 0, 0, 0, 0, 37, 126, 179, 73, 143, 0, 0, 0, 150, 0, 103, 0, 0, 0, 74, 111, 0, 69, 0, 0, 33, 189, 79, 57, 0, 23, 14, 0, 255, 13, 16, 0, 0, 0, 0, 11, 89, 214, 0, 72, 0, 0, 97, 0, 130, 127, 0, 83, 255, 243, 82, 0, 0, 217, 0, 199, 159, 0, 227, 0, 0, 52, 61, 0, 49, 0, 0, 154, 45, 2, 16, 0, 0, 45, 0, 114, 150, 99, 255, 182, 3, 0, 0, 0, 70, 0, 227, 15, 127, 0, 0, 27, 22, 255, 103, 0, 82, 0, 62, 0, 0, 0, 54, 180, 31, 255, 20, 0, 0, 100, 0, 0, 0, 255, 0, 182, 0, 184, 0, 0, 39, 110, 164, 98, 0, 0, 85, 0, 0, 189, 204, 0, 0, 0, 0, 0, 16, 0, 118, 255, 0, 0, 0, 0, 180, 0, 255, 0, 0, 204, 159, 113, 0, 0, 0, 0, 197, 0, 0, 0, 232, 0, 48, 0, 127, 255, 103, 0, 0, 41, 0, 4, 133, 0, 44, 79, 0, 0, 0, 68, 105, 13, 184, 0, 4, 6, 0, 25, 0, 198, 0, 47, 167, 0, 84, 255, 255, 47, 0, 0, 0, 10, 212, 237, 127, 1, 0, 152, 65, 0, 0, 0, 142, 0, 37, 0, 127, 0, 0, 0, 4, 0, 0, 0, 133, 30, 189, 0, 0, 119, 0, 105, 255, 0, 106, 237, 0, 200, 255, 86, 56, 0, 0, 0, 0, 30, 0, 182, 255, 0, 173, 120, 0, 0, 0, 0, 151, 154, 0, 84, 250, 0, 0, 0, 56, 0, 53, 0, 63, 80, 112, 0, 100, 141, 120, 0, 97, 84, 100, 190, 255, 0, 0, 13, 134, 94, 0, 0, 255, 0, 0, 0, 0, 150, 0, 255, 104, 0, 227, 0, 141, 122, 0, 0, 0, 194, 0, 0, 0, 55, 0, 255, 255, 83, 124, 94, 24, 0, 166, 0, 0, 0, 0, 136, 131, 0, 226, 0, 0, 0, 0, 0, 137, 0, 255, 56, 0, 57, 0, 0, 0, 0, 197, 0, 51, 0, 0, 0, 78, 150, 70, 0, 64, 0, 133, 0, 77, 0, 91, 255, 0, 110, 248, 0, 87, 120, 0, 0, 0, 206, 97, 89, 0, 0, 137, 0, 62, 25, 255, 0, 0, 0, 35, 39, 0, 161, 176, 94, 0, 83, 0, 9, 0, 0, 164, 255, 71, 0, 0, 0, 0, 27, 253, 0, 0, 255, 0, 255, 0, 0, 0, 1, 244, 0, 0, 167, 231, 0, 41, 157, 0, 150, 139, 0, 69, 41, 0, 21, 43, 0, 22, 60, 0, 0, 0, 0, 35, 0, 150, 0, 252, 17, 0, 0, 0, 255, 0, 255, 110, 0, 36, 243, 0, 0, 0, 204, 0, 79, 255, 107, 0, 156, 127, 0, 141, 0, 0, 0, 196, 0, 140, 0, 207, 0, 0, 0, 0, 0, 0, 0, 24, 50, 68, 2, 0, 88, 0, 18, 255, 159, 136, 0 }
