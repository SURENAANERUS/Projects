    #ifdef MACOS
        .global _my_printf
        putchar: jmp _putchar
        #ifdef STANDALONE
            .global _main
        #endif
    #else 
        .global my_printf
        _putchar: jmp putchar
        #ifdef STANDALONE
            .global main
        #endif
    #endif

    .bss
    # [unitialized static variables]

    .data
    # [initialized static variables]

    .text
    # [constants/strings]
    test:
    .asciz "%f"
    
    inputt:
    .asciz "4"

    inputtt:
    .asciz "Czy"

    idk:
    .asciz "to"
    idkk:
    .asciz "działa?"
    idkkk:
    .asciz "c"
    stackout:
    .asciz "d"

    pushstack:
    .asciz  "e"
 
  

    _main:
     main:
        pushq   %rbp
        movq    %rsp, %rbp
       
        leaq    test(%rip), %rdi
        
        call    my_printf
           
        movq    %rbp, %rsp
        popq    %rbp
        movq    $0, %rax
        ret
    # int my_printf(const char *fmt, ...)
    _my_printf:
    my_printf:
        # Prologue
        pushq   %rbp
        movq    %rsp, %rbp          
    
        pushq   %rdi    
        pushq   %rsi
        pushq   %rdx
        pushq   %rcx
        pushq   %r8
        pushq   %r9                     # Save register Arguments
        
        subq    $32, %rsp               # Reserve space for the counters, 
        movq    $2, -56(%rbp)           # counter = 2 since starting from rsi
        movq    %r12, -64(%rbp)         # save r12 value on the stack
        movq    $0, -72(%rbp)           # 0 for stack alignemnt
        movq    %rbx, -80(%rbp)         # save the rbx value on the stack

        movq    $2, %r12                # counter = 2 since the return adress is below rbp

        movq    %rdi, %rsi              # load the adress of the format string into rsi for lodsb
        movb    $0, %cl                 # cl = 1 if previous char was %
        
        main_loop:
            lodsb                       # load the chars of the string until we reach the null character
            addb    $0, %al             # turn dil into ascii encoding of a character   
            cmpb    $0, %al
            je      end_loop
            movb    %al, %dil
            cmpb    $1, %cl
            jne     flag_not_set        # jump if last character wasn't a %

            cmpb    $100, %dil          # check if char = 'd'    
            je      print_signed

            cmpb    $117, %dil          # check if char = 'u'   
            je      print_unsigned
            
            cmpb    $115, %dil          # check if char = 's'  
            je      print_string

            cmpb    $37, %dil           # check if char = '%' (after another %)
            jne     not_recognized

            pushq   %rdi                # save these before putchar call
            pushq   %rsi

            call putchar                # print %

            popq    %rsi
            popq    %rdi    
            movb    $0, %cl             # turn the flag 0 again

            jmp     main_loop
            not_recognized:

            pushq   %rdi                
            pushq   %rsi                # save these before putchar call

            movb    $37, %dil
            call    putchar             # print %

            popq    %rsi                
            popq    %rdi                
                                       
            pushq   %rsi
            pushq   $0
            call    putchar             # print the unrecognized char after %
            popq    %rsi
            popq    %rsi
            movb    $0, %cl
            jmp     main_loop           # return to the main loop


            flag_not_set:
                movb    $0, %cl         # for safety reasons make 0 zero if we know it's not 1
                cmpb    $37, %dil       # check if char = '%'
                jne     not_per
                movb    $1, %cl         # set cl = 1
                jmp     main_loop       

                not_per:
                    pushq   %rsi        
                    pushq   %rdi
                    call    putchar     # print the (not)format specifier char
                    popq    %rdi
                    popq    %rsi
                    jmp     main_loop   # return to main loop

        print_signed:   
        cmpq    $7, -56(%rbp)           # check if all register arguments were used
        je      beyond_signed           
        movq    -56(%rbp), %rdx         # move the reg. arg. counter to rdx for the offset calculation
        addq    $1, -56(%rbp)

        salq    $3, %rdx                # left shift the rdx by 3 bits = multiply by 8
        negq    %rdx                    # negate the offset
        addq    %rbp, %rdx              # add rbp to complete the offset

        movslq    (%rdx), %rdi          # move the argument number into rdi(we use movslq to move 32 bit value into 64 bit register)
        cmpq    $0, %rdi                # check if num is negative
        jns     not_neg
        movl    $-1, %ecx               # initialize counter for the number printing loop
        negq    %rdi
        pushq   %rsi
        pushq   %rdi


        movq    $45, %rdi
        call    putchar
    
        popq    %rdi
        popq    %rsi        # TU DOBRZE

        pushq   %rsi
        pushq   %rdi
        jmp     loop_size_signed


        not_neg:
        pushq   %rsi         # save the remaining format string  on the stack(restore it after size loop)
        pushq   %rdi         # save the original argument number on the stack(restore it after size loop)
        movl    $-1, %ecx
        jmp     loop_size_signed

        beyond_signed:
            movq    %r12, %rdx
            incq    %r12
            salq    $3, %rdx        # esentially do the same as above,  but use a different counter(r12) and add to rbp, not substract
            addq    %rbp, %rdx

            movq    (%rdx), %rdi

            cmpq    $0, %rdi     # check if num is negative
            jns     not_neg

            negq    %rdi
            pushq   %rsi
            pushq   %rdi
            pushq   %rcx
            pushq   $0

            movq    $45, %rdi
            call    putchar
            popq    %rcx
            popq    %rcx
            popq    %rdi
            popq    %rsi        # TU DOBRZE

            pushq   %rsi
            pushq   %rdi

            
            movl    $-1, %ecx

        loop_size_signed:
            cmpl    $0, %edi    # check if the digit is equal to 0
            je      end_loop_size_signed
            xor     %edx, %edx  # needed clearing of the most siginifant part of the dividend(needed)
            movl    $10, %esi   # move divider to esi, 32 bit portion of rsi(arbitrary reg. choice)
            movl    %edi, %eax  # move the number to the dividend position required by division
            divl    %esi        # divide the number by 10 and store it in eax
            movl    %eax, %edi  # move back the result into edi
            incl    %ecx        # increment the number of loops performed(size of the digit)
            jmp     loop_size_signed
                                # now ecx holds the number of the digit # CORRECT
        end_loop_size_signed:
        popq    %rdi            # restore rdi, the og number
        
        loop_print_num_sign: # ecx to i, starting at size of digit
            cmpl    $-1, %ecx
            je      end_loop_print_s
            movq    %rcx, %rax          # move the size of the digit(future ending condition) to rax
            pushq   %rcx                # save the orig iterator on the stack

            
            movl    $1, %edx        # set the multiplier
            movl    %ecx, %ebx      # copy the og iterator, to ebx, for using it in the power function TU COŚ ZŁEGO Z ECX JEST
            cmp     $0, %ebx
            jne     power          # check if exponent = 0, if not, go to power. if yes, skip it and make divisor 1
            movq    $1, %rdx
            jmp     end_loop_power_s
          
            power_s:
                imul    $10, %edx 
                decl    %ebx
                jnz     power_s   

            end_loop_power_s:
                
                pushq   %rdi
                pushq   $0      # stack alignemnt
              
                movq    %rdi, %rax      # move the dividend(number) into rax for the div instruction
                movl    %edx, %esi      # move the divisor(10 to the power of ecx) to esi for the div instruction
                xor     %rdx, %rdx      # clear the rdx, it's used to store the remainder
                divl     %esi            # divide, store the value in eax 
                movl    $10, %esi
                xor     %rdx, %rdx
                divl    %esi

                
                movq    %rdx, %rdi      # move the result into rdi for the putchar
                addq    $48, %rdi
                
                call putchar           
                addq    $8, %rsp
                popq    %rdi
                popq    %rcx    # restore the og iterator 
                decl    %ecx    # decrement it
                jmp     loop_print_num_sign


        end_loop_print_s:
        popq    %rsi    # restore the lodsb adress
        movb    $0, %cl # make the % flag 0 again
        jmp main_loop
        print_unsigned:
       
        cmpq    $7, -56(%rbp)
        je      beyond_rbp      # to do
        movq    -56(%rbp), %rdx     # move the offset into rdx CORRECT
        addq    $1, -56(%rbp)       

        salq    $3, %rdx    # left shift the rdx by 3 bits = multiply by 8
        negq    %rdx        # negate the offset
        addq    %rbp, %rdx  # add rbp to complete the offset CORRECT!

        movslq    (%rdx), %rdi # move the argument number into rdi

        pushq   %rsi        # save value of main iterator
        pushq   %rdi        # save og value of number

    

        addq    $48, %rdi   # turn rdi into ascii
        cmpq    $48, %rdi    # check if rdi is 0. If yes, print it
        jne     not_zero
        call    putchar        # print 0

        popq    %rdi
        jmp     end_loop_print  # return to main loop
        not_zero:
        popq    %rdi
        popq    %rsi

        pushq   %rsi    # save the remaining format string  on the stack(restore it after size loop)
        pushq   %rdi    # save the original argument number on the stack(restore it after size loop)
        movl    $-1, %ecx
        jmp     loop_size

        beyond_rbp:
            movq    %r12, %rdx

            salq    $3, %rdx        # esentially do the same as above,  but use a different counter(r12) and add to rbp, not substract
            addq    %rbp, %rdx

            movq    (%rdx), %rdi
            pushq   %rsi
            pushq   %rdi
            movl    $-1, %ecx
            incq    %r12
            


        loop_size:
            cmpl    $0, %edi    # check if the digit is equal to 0
            je      end_loop_size
            xor     %edx, %edx  # needed clearing of the most siginifant part of the dividend(needed)
            movl    $10, %esi   # move divider to esi, 32 bit portion of rsi(arbitrary reg. choice)
            movl    %edi, %eax  # move the number to the dividend position required by division
            divl    %esi        # divide the number by 10 and store it in eax
            movl    %eax, %edi  # move back the result into edi
            incl    %ecx        # increment the number of loops performed(size of the digit)
            jmp     loop_size
                                # now ecx holds the number of the digit # CORRECT
        end_loop_size:
        popq    %rdi            # restore rdi
        
        loop_print_num: # ecx to i, starting at size of digit
            cmpl    $-1, %ecx
            je      end_loop_print
            movq    %rcx, %rax          # move the size of the digit(future ending condition) to rax
            pushq   %rcx                # save the orig iterator on the stack

            
            movl    $1, %edx        # set the multiplier
            movl    %ecx, %ebx      # copy the og iterator, to ebx, for using it in the power function TU COŚ ZŁEGO Z ECX JEST
            cmp     $0, %ebx
            jne      power          # check if exponent = 0, if not, go to power. if yes, skip it and make divisor 1
            movq    $1, %rdx
            jmp     end_loop_power
          
            power:
                imul    $10, %edx 
                decl    %ebx
                jnz     power   

            end_loop_power:
                
                pushq   %rdi
                pushq   $0      # stack alignemnt
              
                movq    %rdi, %rax      # move the dividend(number) into rax for the div instruction
                movl    %edx, %esi      # move the divisor(10 to the power of ecx) to esi for the div instruction
                xor     %rdx, %rdx      # clear the rdx, it's used to store the remainder
                divl    %esi            # divide, store the value in eax 
                movl    $10, %esi
                xor     %rdx, %rdx
                divl    %esi

                
                movq    %rdx, %rdi      # move the result into rdi for the putchar
                addq    $48, %rdi
                
                call putchar           
                addq    $8, %rsp
                popq    %rdi
                popq    %rcx    # restore the og iterator 
                decl    %ecx    # decrement it
                jmp     loop_print_num


        end_loop_print:
        popq    %rsi    # restore the lodsb adress
        movb    $0, %cl # make the % flag 0 again
        jmp main_loop

        print_string:   # to do
            cmpq    $7, -56(%rbp)       # check number of arguments 
            je      beyond_string
            movq    -56(%rbp), %rdx     
            addq    $1, -56(%rbp)

            salq    $3, %rdx    # left shift the rdx by 3 bits = multiply by 8
            negq    %rdx        # negate the offset
            addq    %rbp, %rdx  # add rbp to complete the offset

            movq    (%rdx), %rdi # move the argument string, at adress rdx, to rdi
            pushq   %rsi         # save old rsi
          # stack alignemnt
            pushq   $0
            movq    %rdi, %rsi   # move the string ptr to rsi for lodsb

            jmp     loop_string

            beyond_string:
                movq    %r12, %rdx
                incq    %r12
                salq    $3, %rdx        # esentially do the same as above,  but use a different counter(r12) and add to rbp, not substract
                addq    %rbp, %rdx      # argument misaligment!!! ! gets 33 while it should get the 2nd string!! r12 not correctly decremented

                movq    (%rdx), %rdi  
                pushq   %rsi
                pushq   $0
                
                movq    %rdi, %rsi

            loop_string:
                lodsb               # tu nagle rsi - 33??
                addb    $0, %al             # turn al into ascii encoding
                cmpb    $0, %al             #  if end of string, finish printing    # czemu za drugą iteracją nie wykrywa null ptr?
                je      end_loop_string
                movb    %al, %dil
                pushq   %rdi
                pushq   %rsi                # save new rsi
                call    putchar
                popq    %rsi
                popq    %rdi
                jmp     loop_string

            end_loop_string:
                
                popq    %rsi
                popq    %rsi
                movb    $0, %cl # make the % flag 0 again
                jmp     main_loop
        
        end_loop:
            movq    -64(%rbp), %r12
            movq    -80(%rbp), %rbx
            movq    %rbp, %rsp
            popq    %rbp
            ret