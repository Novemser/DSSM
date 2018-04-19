package com.novemser.util

import java.util.concurrent.TimeUnit

import scala.collection.mutable

class Timer {
  private val timerMap: mutable.Map[String, Long] = mutable.Map[String, Long]()
  private val countMap: mutable.Map[String, Long] = mutable.Map[String, Long]()

  def initTimer(name: String): Timer = {
    timerMap(name) = 0
    countMap(name) = 0
    this
  }

  def time[R](block: => R, msg: String, name: String): R = {
    val t0 = System.currentTimeMillis()
    val result = block // call-by-name
    val t1 = System.currentTimeMillis()
    timerMap(name) += (t1 - t0)
    countMap(name) += 1
//    println(s"$msg elapsed time: " + (t1 - t0) + "ns")
    result
  }

  def printTime(): Unit = {
    timerMap foreach { kv =>
      {
        println(s"${kv._1} => Total time:${kv._2}, Average time:${kv._2 / countMap(kv._1)}")
      }
    }
  }

  def reset(): Unit = {
    timerMap.clear()
    countMap.clear()
  }
}
