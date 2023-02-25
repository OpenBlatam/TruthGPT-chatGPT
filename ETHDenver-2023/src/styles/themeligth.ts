// theme.js
import { extendTheme } from '@chakra-ui/react'

// Version 1: Using objects
export const theme2 = extendTheme({
  colors: {
    backgroundDark: '#1F1B24',
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
