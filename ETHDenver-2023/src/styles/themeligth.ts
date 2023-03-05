// theme.js
import { extendTheme } from '@chakra-ui/react'

// Version 1: Using objects
export const theme2 = extendTheme({
  colors: {
    backgroundDark: '#00000A',

  },
  styles: {
    global: {
      body: {
        bg: ' #ffffff',
        color: 'white'
      }
    }
  }
})
