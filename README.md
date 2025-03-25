该项目采集的数据可用于bench2drivezoo项目下
运行示例：
在carla里面运行采集数据的仿真：
python collect.py --map Town01 --weather_preset ClearNoon --filter vehicle.tesla.model3 --sync --res 1600x900 --max_frames 1000
解释：
--map Town01 ：指定要加载的CARLA地图为 Town01。
--filter vehicle.tesla.model3 ：选择特定的汽车模型作为Ego车辆。
--sync ：以同步模式运行仿真。
--res 1600x900 ：设置输出窗口的分辨率为1600x900。
--max_frames 1000 ：设置最大帧数为1000，达到此帧数后结束仿真和数据采集。
--weather_preset ClearNoon：指定天气状况
