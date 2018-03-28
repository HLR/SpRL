package edu.tulane.cs.hetml.nlp.sprl.ontologies
import java.io.{BufferedReader, InputStreamReader}

import scala.sys.process.ProcessBuilder._
import scala.sys.process._
/**
  * Created by parisakordjamshidi on 1/19/18.
  */
object parseWithTrips extends App {

  "echo $TRIPS_BASE" !

  "/usr/local/bin/ruby /Users/parisakordjamshidi/step/src/Systems/STEP/batch.rb --input-file=/Users/parisakordjamshidi/step/TRIPS/Systems/STEP/test.txt -e txt"!

//  "bash /Users/parisakordjamshidi/step/TRIPS/Systems/STEP/batch.rc --input-file=\"/Users/parisakordjamshidi/step/TRIPS/Systems/STEP/test.txt\" -e \"txt\""!

}
