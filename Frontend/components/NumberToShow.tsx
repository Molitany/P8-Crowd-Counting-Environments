import React from 'react'
import { Text } from './Themed'
import { StyleSheet } from 'react-native';


const NumberToShow = (count:number) => {
    return <Text style={styles.countStyle}> This is the count {count}!</Text>

}

// function NumberToShow(count:number){
//     return <Text style={styles.countStyle}> This is the count {count}!</Text>
// }

const styles = StyleSheet.create({
    countStyle: {
      fontSize: 30,
      textAlign: 'center',
      resizeMode:"contain",
    },
})



export default NumberToShow