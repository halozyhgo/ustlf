# encoding = utf-8
from flask import Flask, request
import multiprocessing
from multiprocessing import Pool
import requests
import threading
import json
from datetime import datetime
import mysql.connector
from UniversalEnergyStoragePlanPowerGenerationService.planPowerSolve import Solve

app = Flask(__name__)
encoding_utf8 = 'utf-8'

# 储能动态计划功率生成
@app.route('/energyStoragePlanPower', methods=['POST'])
def spotMarketCalculate():
    # # 记录请求时间
    # start_time = datetime.now()

    # 获取请求参数
    request_data = request.get_json()

    if "url" in request_data:
        if request_data["url"] != "":
            requestFlag = True
        else:
            requestFlag = False
    else:
        requestFlag = False

    # 判断是同步请求还是异步请求
    if requestFlag:
        # 异步
        # 保存请求参数到队列中
        requestQ.put(request_data)
        # 返回请求成功
        code = "200"
        msg = "请求成功"
        data = []
        messageId = request_data["system"]["messageId"]
        if "stdName" in request_data:
            stdName = request_data["stdName"]
        else:
            stdName = "test"
    else:
        # 同步
        try:
            # 生成计划功率
            code, msg, data, messageId, stdName = solve.start(request_data)

        except Exception as e:
            code = "203"
            msg = "计算失败:{}".format(e)
            data = []
            messageId = request_data["system"]["messageId"]
            stdName = ""

    # # 请求结束时间
    # end_time = datetime.now()
    # # 响应时间
    # response_time = (end_time - start_time).total_seconds()
    # # 请求场景
    # responseQ.put([start_time, request_data["scene"], response_time, code])  # 请求时间、请求场景、响应时间、响应状态
    return {"code": code, "msg": msg, "data": data, "messageId": messageId, "stdName": stdName}

# 外部调用接口（中央研究院）
@app.route('/planPower', methods=['POST'])
def planPowerCalculate():
    # 获取请求参数
    request_data = request.get_json()
    # 将请求参数封装成算法服务接口需要的格式
    data = request_data["data"]
    dataT = []
    tempLoad = []

    for i in range(len(data)):
        dateT = data[i]["date"]
        priceT = data[i]["price"]
        loadT = data[i]["load"]
        if loadT != []:
            tempLoad.append(min(loadT))
        dataT.append({"date": dateT, "chargePrice": priceT, "dischargePrice": priceT, "load": loadT, "priceCode": {}, "isWork": True, "type": "", "timeSpan": 24/len(priceT)*60, "demandThreshold": 0})

    systemT = {}
    ratedCapacity = request_data["system"]["ratedCapacity"]
    ratedPower = request_data["system"]["ratedPower"]
    systemT["initialSOC"] = 0
    systemT["downSOC"] = 0
    systemT["upSOC"] = 1
    systemT["ratedCapacity"] = ratedCapacity
    systemT["ratedChargingPower"] = -ratedPower
    systemT["ratedDischargePower"] = ratedPower
    systemT["efIn"] = 0.93
    systemT["efOut"] = 0.93
    systemT["chargeEfficiency"] = []
    systemT["dischargeEfficiency"] = []
    systemT["counterCurrentThreshold"] = 0
    if tempLoad != []:
        systemT["overloadThreshold"] = min(tempLoad) + min(tempLoad) * 0.2
    else:
        systemT["overloadThreshold"] = -ratedPower - ratedPower * 0.2
    systemT["llcoe"] = 0.25
    systemT["messageId"] = ""
    systemT["delay"] = 0
    systemT["cycles"] = 0

    constraint = {}
    constraint["chargePowerConstraint"] = True
    constraint["disChargePowerConstraint"] = True
    constraint["SOCConstraint"] = True
    constraint["overloadConstraint"] = True
    constraint["counterCurrentConstraint"] = True
    constraint["cyclesConstraint"] = False
    constraint["demandConstraint"] = False
    constraint["loadOptimize"] = True

    requestQ = {}
    requestQ["data"] = dataT
    requestQ["system"] = systemT
    requestQ["constraint"] = constraint
    requestQ["scene"] = "powMart"
    requestQ["url"] = ""
    requestQ["continuous"] = True

    response = app.test_client().post('/energyStoragePlanPower', json=requestQ)

    msg = eval(response.text)["msg"]
    code = eval(response.text)["code"]
    dataRes = eval(response.text)["data"]
    resTT = []
    for k in range(len(dataRes)):
        dateRes = dataRes[k]["date"]
        planedPowerRes = dataRes[k]["planedPower"]
        for m in range(len(planedPowerRes)):
            planedPowerRes[m].pop("periodOfTime")
        resTT.append({"date": dateRes, "planedPower": planedPowerRes})

    return {"msg": msg, "code": str(code), "data": resTT}


# 调用数据分析
@app.route('/dataAnalysis', methods=['POST'])
def invokeDataAnalysis():
    try:
        # 获取请求参数
        request_data = request.get_json()
        startTime = request_data["startTime"]
        endTime = request_data["endTime"]
        # 连接数据库
        cnx = mysql.connector.connect(
            host=host,
            user=user,
            port=port,
            password=password,
            database=database
        )
        # 创建游标
        cursor = cnx.cursor()
        # 执行查询所有数据的SQL语句
        query = "SELECT * FROM invoke_data where start_time >= %s AND start_time <= %s"
        cursor.execute(query, (startTime, endTime))
        # 获取查询结果
        results = cursor.fetchall()
        # 关闭游标和连接
        cursor.close()
        cnx.close()
        # 查询结果
        # 时段内总调用次数
        allInvokeNums = len(results)

        # 各场景调用次数
        powMartNums = 0
        powTradeNums = 0
        iSolarBPNums = 0
        otherNums = 0
        # 响应时间
        responseTime = 0
        # 响应成功次数
        responseSu = 0

        for row in results:
            # 统计响应时间
            responseTime += row[3]
            # 统计场景调用次数
            if row[2] == "powMart":
                powMartNums += 1
            elif row[2] == "powTrade":
                powTradeNums += 1
            elif row[2] == "iSolarBP":
                iSolarBPNums += 1
            else:
                otherNums += 1
            # 统计响应成功次数
            if row[4] == "200":
                responseSu += 1
        code = "200"
        msg = "请求成功"
        data = {"startTime": startTime, "endTime": endTime, "totalNumberOfCalls": allInvokeNums, "powMartCalls": powMartNums,
                "powTradeCalls": powTradeNums, "iSolarBPCalls": iSolarBPNums, "otherCalls": otherNums,
                "averageResponseTime": responseTime/allInvokeNums, "responseSuccessPower": responseSu/allInvokeNums}
    except Exception as e:
        code = "204"
        msg = "数据库读取失败:{}".format(e)
        data = []
    return {"code": code, "msg": msg, "data": data}

# 异步请求处理
def asynchronousRequestProcess(requestQ):
    while True:
        # 创建进程池
        with Pool(processes=10) as pool:
            # print("请求队列大小:", q.qsize())
            request_data = requestQ.get()
            try:
                # 将任务分配给进程池
                tempData = pool.map(solve.start, [request_data])
                code = tempData[0][0]
                msg = tempData[0][1]
                data = tempData[0][2]
                messageId = tempData[0][3]

                response = requests.post(request_data["url"], json={"code": code, "msg": msg, "data": data, "messageId": messageId})
                # print("状态码:", response.status_code)
                # print("响应内容:", response.text)

            except Exception as e:
                code = "202"
                msg = "回调失败:{}".format(e)
                data = []
                messageId = request_data["system"]["messageId"]
                response = requests.post(request_data["url"], json={"code": code, "msg": msg, "data": data, "messageId": messageId})
                # print("状态码:", response.status_code)
                # print("响应内容:", response.text)

# 请求响应数据存储
def responseDataStorage(responseQ):
    # 连接数据库
    cnx = mysql.connector.connect(
        host=host,
        user=user,
        port=port,
        password=password,
        database=database
    )
    # 创建游标
    cursor = cnx.cursor()
    while True:
        resData = responseQ.get()
        try:
            # 插入数据
            insert_query = "INSERT INTO invoke_data (start_time, scene, response_time, code) VALUES (%s, %s, %s, %s)"
            data = (resData[0].strftime('%Y-%m-%d %H:%M:%S'), resData[1], resData[2], resData[3])
            cursor.execute(insert_query, data)
            # 提交事务
            cnx.commit()
        except Exception as e:
            print("数据导入数据库失败:{}".format(e))

# 创建对象
solve = Solve()

# 创建请求队列
requestQ = multiprocessing.Queue()

# 请求响应数据
responseQ = multiprocessing.Queue()

# 读取配置文件
with open('./config.json', 'r') as f:
    config = json.load(f)
host = config["databaseInfo"]["host"]
user = config["databaseInfo"]["user"]
port = config["databaseInfo"]["port"]
password = config["databaseInfo"]["password"]
database = config["databaseInfo"]["database"]
userCenter = config["userCenter"]

if __name__ == '__main__':
    # 开启异步请求监控线程
    monitor_thread1 = threading.Thread(target=asynchronousRequestProcess, args=(requestQ,))
    monitor_thread1.start()
    # # 开启请求响应数据存储线程
    # monitor_thread2 = threading.Thread(target=responseDataStorage, args=(responseQ,))
    # monitor_thread2.start()
    # 开启服务
    app.run(host="0.0.0.0", port=18009, processes=True)
