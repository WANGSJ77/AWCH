import os
import random

class SplitFiles():
    """按行分割文件"""

    def __init__(self, file_name):
        """初始化要分割的源文件名和分割后的文件行数"""
        self.file_name = file_name

    def get_random(self):
        """生成随机数组，随机划分 （0，190001）txt标签行数， 7600测试集标签行数"""
        random_num = random.sample(range(0, 2100), 420)

        return random_num

    def split_file(self):
        r = self.get_random()
        if self.file_name and os.path.exists(self.file_name):
            try:
                with open(self.file_name) as f:  # 使用with读文件
                    temp_count = 1
                    for line in f:
                        if temp_count in r:
                            self.write_file('test', line)
                        else:
                            self.write_file('train', line)
                        temp_count += 1

            except IOError as err:
                print(err)
        else:
            print("%s is not a validate file" % self.file_name)

    def get_part_file_name(self, part_name):
        """"获取分割后的文件名称：在源文件相同目录下建立临时文件夹temp_part_file，然后将分割后的文件放到该路径下"""
        temp_path = os.path.dirname(self.file_name)  # 获取文件的路径（不含文件名）
        part_file_name = temp_path + "/UCM_" + str(part_name) + "_list.txt"
        return part_file_name

    def write_file(self, part_num, line):
        """将按行分割后的内容写入相应的分割文件中"""
        part_file_name = self.get_part_file_name(part_num)
        try:
            with open(part_file_name, "a") as part_file:
                part_file.writelines(line)
        except IOError as err:
            print(err)


if __name__ == "__main__":
    file = SplitFiles(r"../Data/UCM/UCM.txt")
    file.split_file()
