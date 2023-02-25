// theme.js
import { extendTheme } from '@chakra-ui/react'

// Version 1: Using objects
export const theme = extendTheme({
  colors: {
    backgroundDark: '#AEAEBB',
  },
  styles: {
    global: {
      body: {
        bg: '#AEAEBB',
        color: 'white'
      }
    }
  }
})
